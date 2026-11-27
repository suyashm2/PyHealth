"""A PyHealth 2.1 compatible SHy starting point for next-visit diagnosis prediction.

The model builds hierarchy-aware initial code embeddings, applies a batched
UniGIN-style code -> visit -> code message-passing step on patient-isolated
hypergraphs, augments likely missing diagnosis-visit links, extracts
phenotype-specific sub-hypergraphs, and embeds each phenotype over time with a
GRU and visit attention, then weights phenotype-specific diagnosis predictions.
The training objective combines diagnosis prediction with the paper's
hypergraph-reconstruction fidelity, phenotype distinctness, and phenotype
importance regularization losses.

Example task sample before PyHealth converts codes to token IDs::

    {
        "patient_id": "P1",
        "visit_id": "103",  # The visit being predicted; one ID, not visit history.
        "conditions": [
            ["4280", "25000"],  # Historical visit 1
            ["V4581"],            # Historical visit 2
        ],
        "next_conditions": ["4280", "41401"],  # Diagnoses at the next visit
    }

After task processing and batching, ``forward`` receives numeric tensors. For a
batch of two patients, the shapes may look like::

    conditions:      [2, 2, 2]  # [patients, visits, code slots]
    next_conditions: [2, 3]     # [patients, label-vocabulary entries]

Input code IDs and label positions may use separate vocabularies. The hierarchy
embedding rows are aligned to the fitted input vocabulary used by ``conditions``.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.medcode import InnerMap
from pyhealth.models import BaseModel


class SHyRepro(BaseModel):
    """Predict multi-label next-visit diagnoses from nested diagnosis histories.

    SHy (Yu et al., CHIL 2025) builds a hypergraph for each patient, with
    diagnosis codes as nodes and visits as hyperedges. It adds likely missing
    code-visit links, extracts K temporal phenotypes (subgraphs), and predicts
    the next visit's diagnoses from a weighted mix of per-phenotype
    predictions. Paper: https://proceedings.mlr.press/v287/yu25a.html

    Args:
        dataset: Fitted sample dataset with one nested diagnosis-code input
            (for example ``conditions``) and one multilabel next-visit label.
        embedding_dim: Width of each diagnosis-code embedding. Must be
            divisible by ``num_hierarchy_levels``. Default 64.
        hidden_dim: Width of the message-passing, phenotype-mask, GRU and
            attention layers. ``None`` uses ``embedding_dim``. Default None.
        num_hierarchy_levels: Number of ICD-9 hierarchy levels whose
            embeddings are concatenated into each code embedding (chapter,
            section, category, full code). Only 4 is supported. Default 4.
        num_message_passing_layers: Number of UniGIN message-passing layers
            over each patient's hypergraph. Default 2.
        augmentation_ratio: Fraction of each patient's existing code-visit
            links to add as likely missing links before phenotype extraction.
            0 turns augmentation off. Default 0.2.
        num_similarity_heads: Number of random projection heads used to score
            candidate missing links during augmentation. Default 10.
        num_phenotypes: Number of temporal phenotypes K extracted per
            patient. Default 5.
        phenotype_temperature: Gumbel-softmax temperature for the phenotype
            masks. Default 1.0.
        sample_phenotypes_at_eval: If True, sample random phenotype masks at
            evaluation as in training (the authors' code). If False,
            threshold the mask probabilities at 0.5 instead. Default True.
        phenotype_code_chunk_size: Number of codes scored at once when
            building phenotype masks; lower values use less memory and do not
            change results. Default 128.
        phenotype_attention_dim: Width of the self-attention over phenotype
            embeddings. Must be divisible by ``num_phenotype_attention_heads``.
            Default 256.
        num_phenotype_attention_heads: Number of heads in the phenotype
            self-attention. Default 8.
        prediction_activation: ``"softmax"`` (the paper) or ``"sigmoid"`` for
            the per-phenotype diagnosis predictions. Default ``"softmax"``.
        fidelity_loss_weight: Weight of the loss for rebuilding the original
            hypergraph from the phenotypes. Default 0.003.
        distinct_loss_weight: Weight of the loss that keeps phenotypes
            different from each other. Default 0.00025.
        alpha_loss_weight: Weight of the regularizer that keeps the phenotype
            mixing weights from collapsing onto one phenotype. Default 0.04.
        embedding_init_std: Standard deviation of the normal distribution
            used to initialize the hierarchy embedding tables. Default 1.0,
            as in the authors' code.

    Raises:
        ValueError: If a setting is out of range, or if the dataset does not
            have exactly one input feature and one label.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import SHyRepro
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-2",
        ...         "conditions": [["4019", "25000"], ["4280", "4019"]],
        ...         "next_conditions": ["4280", "42731"],
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": [["486", "5849"]],
        ...         "next_conditions": ["486"],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "nested_sequence"},
        ...     output_schema={"next_conditions": "multilabel"},
        ...     dataset_name="test",
        ... )
        >>> model = SHyRepro(dataset=dataset, embedding_dim=32)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> output = model(**next(iter(loader)))
        >>> output["y_prob"].shape  # 2 patients, 3 possible next-visit codes
        torch.Size([2, 3])
        >>> output["loss"].backward()  # total of SHy's four training losses
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 64,
        hidden_dim: Optional[int] = None,
        num_hierarchy_levels: int = 4,
        num_message_passing_layers: int = 2,
        augmentation_ratio: float = 0.2,
        num_similarity_heads: int = 10,
        num_phenotypes: int = 5,
        phenotype_temperature: float = 1.0,
        sample_phenotypes_at_eval: bool = True,
        phenotype_code_chunk_size: int = 128,
        phenotype_attention_dim: int = 256,
        num_phenotype_attention_heads: int = 8,
        prediction_activation: str = "softmax",
        # Authors' reported MIMIC-III/IV objective weights (README table).
        fidelity_loss_weight: float = 0.003,
        distinct_loss_weight: float = 0.00025,
        alpha_loss_weight: float = 0.04,
        # Authors' code: nn.Embedding default init, N(0, 1).
        embedding_init_std: float = 1.0,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError("SHy currently expects exactly one nested code feature")
        if len(self.label_keys) != 1:
            raise ValueError("SHy currently expects exactly one label key")
        if num_hierarchy_levels != 4:
            raise ValueError(
                "SHy's ICD-9 hierarchy uses exactly four levels: chapter, "
                "section, category, and full diagnosis code"
            )
        if embedding_dim % num_hierarchy_levels != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_hierarchy_levels so each "
                "hierarchy level has the same embedding width"
            )
        if num_message_passing_layers <= 0:
            raise ValueError("num_message_passing_layers must be positive")
        if augmentation_ratio < 0:
            raise ValueError("augmentation_ratio must be non-negative")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        if num_similarity_heads <= 0:
            raise ValueError("num_similarity_heads must be positive")
        if num_phenotypes <= 0:
            raise ValueError("num_phenotypes must be positive")
        if phenotype_temperature <= 0:
            raise ValueError("phenotype_temperature must be positive")
        if phenotype_code_chunk_size <= 0:
            raise ValueError("phenotype_code_chunk_size must be positive")
        if phenotype_attention_dim <= 0:
            raise ValueError("phenotype_attention_dim must be positive")
        if num_phenotype_attention_heads <= 0:
            raise ValueError("num_phenotype_attention_heads must be positive")
        if phenotype_attention_dim % num_phenotype_attention_heads != 0:
            raise ValueError(
                "phenotype_attention_dim must be divisible by "
                "num_phenotype_attention_heads"
            )
        prediction_activation = prediction_activation.lower()
        if prediction_activation not in {"softmax", "sigmoid"}:
            raise ValueError("prediction_activation must be 'softmax' or 'sigmoid'")
        if min(fidelity_loss_weight, distinct_loss_weight, alpha_loss_weight) < 0:
            raise ValueError("SHy explanation-loss weights must be non-negative")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        # BaseModel resolves the label processor to the plain task mode
        # (e.g. "multilabel"); keep that normalized value for its loss helpers.
        if prediction_activation == "sigmoid" and self.mode == "multiclass":
            raise ValueError(
                "The sigmoid prediction ablation is for multilabel diagnosis targets, "
                "not multiclass targets"
            )
        self.embedding_dim = embedding_dim
        self.num_hierarchy_levels = num_hierarchy_levels
        self.per_level_dim = embedding_dim // num_hierarchy_levels
        self.augmentation_ratio = float(augmentation_ratio)
        self.num_similarity_heads = num_similarity_heads
        self.num_phenotypes = num_phenotypes
        self.phenotype_temperature = float(phenotype_temperature)
        # The authors' released code samples Gumbel masks at test time too, so
        # evaluation sees the same kind of masks as training. False switches to
        # deterministic thresholding at probability 0.5 during evaluation.
        self.sample_phenotypes_at_eval = bool(sample_phenotypes_at_eval)
        self.phenotype_code_chunk_size = phenotype_code_chunk_size
        self.phenotype_attention_dim = phenotype_attention_dim
        self.num_phenotype_attention_heads = num_phenotype_attention_heads
        self.prediction_activation = prediction_activation
        self.fidelity_loss_weight = float(fidelity_loss_weight)
        self.distinct_loss_weight = float(distinct_loss_weight)
        self.alpha_loss_weight = float(alpha_loss_weight)
        hidden_dim = hidden_dim or embedding_dim
        self.hidden_dim = hidden_dim

        # ---------------------------------------------------------------------
        # STEP 1 — Recover PyHealth's exact input-code <-> integer-ID mapping.
        # 1a. Read the vocabulary fitted for the `conditions` input field.
        # 1b. Invert it by ID, so row i in M will correspond to input token ID i.
        # 1c. Read special-token IDs from the processor instead of guessing them.
        # ---------------------------------------------------------------------
        input_processor = self.dataset.input_processors[self.feature_key]
        code_vocab = getattr(input_processor, "code_vocab", None)
        if not isinstance(code_vocab, dict) or not code_vocab:
            raise TypeError(
                "SHy needs the fitted input processor's `code_vocab` mapping "
                "(diagnosis token -> integer ID) to align M with H"
            )
        self.num_nodes = input_processor.vocab_size()
        id_to_token: List[Optional[str]] = [None] * self.num_nodes
        for token, token_id in code_vocab.items():
            token_id = int(token_id)
            if not 0 <= token_id < self.num_nodes:
                raise ValueError(f"Input token ID {token_id} is outside processor size")
            if id_to_token[token_id] is not None:
                raise ValueError(f"Two input tokens map to ID {token_id}")
            id_to_token[token_id] = str(token)
        if any(token is None for token in id_to_token):
            raise ValueError("Input processor vocabulary IDs must be contiguous from 0")
        self.id_to_code = tuple(id_to_token)
        self.pad_idx = int(code_vocab.get("<pad>", 0))
        self.unk_idx = int(code_vocab.get("<unk>", 1))

        # ---------------------------------------------------------------------
        # STEP 2 — Resolve each input diagnosis to SHy's four semantic levels.
        # 2a. Standardize strings such as "4280" to the hierarchy's "428.0".
        # 2b. Read PyHealth's ancestors and identify the broad chapter/section
        #     ranges, then derive the category and retain the complete code.
        # 2c. Remove only the universal numeric root. V/E range roots remain as
        #     their real chapter nodes. Ignore intermediate decimal subcodes.
        # 2d. Assign a level-local integer ID to each node. Special tokens get
        #     their own IDs and are masked from graph operations.
        # ---------------------------------------------------------------------
        icd9cm = InnerMap.load("ICD9CM")
        code_level_node_ids, level_vocab_sizes, unmapped = self._make_level_node_ids(
            self.id_to_code, icd9cm, num_hierarchy_levels
        )
        if unmapped:
            warnings.warn(
                f"{len(unmapped)} input tokens were not found in the ICD-9-CM "
                "hierarchy. Following the SHy preprocessing, they share the "
                "all-zero hierarchy path.",
                stacklevel=2,
            )
        self.register_buffer("code_level_node_ids", code_level_node_ids)

        # ---------------------------------------------------------------------
        # STEP 3 — Create the learnable hierarchy tables and assemble initial M.
        # 3a. Each level gets its own learnable embedding table, shared by codes
        #     that have the same ontology node at that level.
        # 3b. _initial_code_embeddings() looks up one vector per level and
        #     concatenates them, yielding M with shape [input codes, embedding_dim].
        # ---------------------------------------------------------------------
        self.hierarchy_embeddings = nn.ModuleList(
            [nn.Embedding(size, self.per_level_dim) for size in level_vocab_sizes]
        )
        for table in self.hierarchy_embeddings:
            nn.init.normal_(table.weight, mean=0.0, std=embedding_init_std)

        # ---------------------------------------------------------------------
        # STEP 4 — Set up patient-specific UniGIN-style message passing.
        # 4a. H chooses which code rows contribute to each visit.
        # 4b. Each layer averages code vectors into visits, then sums those visit
        #     vectors back into codes while retaining each code's own state.
        # 4c. Parameters are shared across patients; batched H keeps their data
        #     and updated code states separate.
        # ---------------------------------------------------------------------
        self.message_layers = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim)
                for _ in range(num_message_passing_layers)
            ]
        )
        self.message_eps = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(num_message_passing_layers)]
        )
        self.message_activation = nn.LeakyReLU()

        # STEP 5 — Learn the feature-wise weights used to score missing links.
        # Each row is one weighted-cosine head Phi_k from the SHy paper.
        # The paper's selected setup uses n_s=10 heads and p=0.2.
        self.similarity_projections = nn.Parameter(
            torch.randn(num_similarity_heads, embedding_dim)
        )

        # STEP 6 — Create one independent diagnosis-visit mask scorer per phenotype.
        # 6a. Each scorer reads the concatenated code and visit embeddings.
        # 6b. Its scalar output is a Bernoulli logit for retaining that edge.
        # 6c. extract_phenotypes uses hard-forward straight-through Gumbel-Sigmoid.
        self.phenotype_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_phenotypes)
            ]
        )

        # STEP 7 — Turn each phenotype's selected visits into a temporal vector.
        # The GRU is shared across patients and phenotype IDs; each sequence is
        # still distinct because its masked visit embeddings come from its own Ψ.
        self.phenotype_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # STEP 8 — Let phenotypes exchange information, then estimate alpha.
        # These defaults match the authors' released setup: key_dim=256 and
        # eight self-attention heads. The explicit projections correspond to
        # W_Q, W_K, and W_V; MultiheadAttention computes Equation 13 per head.
        self.phenotype_query_projection = nn.Linear(hidden_dim, phenotype_attention_dim)
        self.phenotype_key_projection = nn.Linear(hidden_dim, phenotype_attention_dim)
        self.phenotype_value_projection = nn.Linear(hidden_dim, phenotype_attention_dim)
        self.phenotype_self_attention = nn.MultiheadAttention(
            embed_dim=phenotype_attention_dim,
            num_heads=num_phenotype_attention_heads,
            batch_first=True,
        )
        # Equation 14 maps each context-aware phenotype vector to one score.
        self.phenotype_importance = nn.Linear(phenotype_attention_dim, 1, bias=False)

        # STEP 9 — Predict diagnoses independently from every phenotype (Eq. 15).
        # The same diagnosis classifier is applied to each U_k, then the per-
        # phenotype predictions are mixed using alpha. `prediction_activation`
        # selects the paper's softmax or the sigmoid ablation.
        self.diagnosis_classifier = nn.Linear(hidden_dim, self.get_output_size())

        # STEP 10 — Reconstruct the original [code, visit] patient graph.
        # Appendix C concatenates U_1..U_K and repeats that same context vector
        # as the GRU input at every visit step. The resulting hidden state for
        # each step is projected to code probabilities.
        self.hypergraph_decoder_gru = nn.GRU(
            input_size=num_phenotypes * hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.hypergraph_decoder_output = nn.Linear(hidden_dim, self.num_nodes)

    @staticmethod
    def _make_level_node_ids(
        id_to_code: Sequence[Optional[str]], icd9cm: Any, num_levels: int
    ) -> Tuple[torch.Tensor, List[int], List[str]]:
        """Map processor IDs to [chapter, section, category, full-code] nodes.

        PyHealth returns ancestors closest-first. We reverse them and retain the
        first two non-universal range nodes as chapter and section. Intermediate
        decimal nodes (for example ``250.0`` on the way to ``250.00``) are not
        used as hierarchy levels; the category is the code's ICD-9 category
        prefix, and the final level is always the complete standardized code.

        As in the SHy preprocessing, hierarchy ID 0 is reserved at every level.
        Codes missing from the ICD-9 range map receive [0, 0, 0, 0]; all known
        hierarchy nodes receive positive IDs.
        """
        if num_levels != 4:
            raise ValueError("ICD-9 hierarchy paths must have exactly four levels")

        # SHy's preprocessing uses all-zero IDs for codes absent from its ICD-9
        # range file. Reserve ID 0 in every table for that shared fallback.
        level_vocabularies: List[Dict[str, int]] = [
            {"__unmapped__": 0} for _ in range(num_levels)
        ]
        rows: List[List[int]] = []
        unmapped: List[str] = []
        special_tokens = {"<pad>", "<unk>"}

        for raw_token in id_to_code:
            token = str(raw_token)
            if token in special_tokens:
                # PyHealth's padding/unknown tokens are not diagnoses. Keep them
                # in the same reserved zero slot used by SHy's preprocessing.
                path = ["__unmapped__"] * num_levels
            else:
                try:
                    standardized = icd9cm.standardize(token)
                    if standardized not in icd9cm:
                        raise KeyError(f"{standardized} is not in ICD9CM")
                    # Ancestors arrive nearest-first. Reversing makes broad ICD
                    # ranges appear before narrower ranges in the hierarchy.
                    ancestors = list(icd9cm.get_ancestors(standardized))
                    root_first = list(reversed(ancestors))

                    # 001-999.99 is the dataset-wide numeric root, not a disease
                    # chapter. Remove it wherever it occurs. V/E family ranges
                    # (such as V01-V91.99 or E000-E999.99) are retained: they are
                    # meaningful chapter-level nodes for those code families.
                    range_nodes = [
                        node
                        for node in root_first
                        if "-" in node and node != "001-999.99"
                    ]

                    # The paper's four slots are Chapter -> Section -> Category
                    # -> Full code. If a code path has no explicit section range,
                    # use a shared placeholder under its chapter.
                    if not range_nodes:
                        raise KeyError(f"No ICD-9 chapter range for {standardized}")
                    chapter = range_nodes[0]
                    section = (
                        range_nodes[1]
                        if len(range_nodes) > 1
                        else f"__no_section__:{chapter}"
                    )

                    # ICD-9 numeric and V codes use a 3-character category
                    # (e.g. 250 or V45); E codes use E plus three digits (E812).
                    category_width = 4 if standardized.upper().startswith("E") else 3
                    category = standardized[:category_width]
                    path = [chapter, section, category, standardized]
                except (KeyError, ValueError, TypeError):
                    # Match SHy's released preprocessing: unmapped codes use
                    # [0, 0, 0, 0], rather than code-specific virtual nodes.
                    path = ["__unmapped__"] * num_levels
                    unmapped.append(token)

            row_ids = []
            for level, hierarchy_node in enumerate(path):
                level_vocab = level_vocabularies[level]
                if hierarchy_node not in level_vocab:
                    level_vocab[hierarchy_node] = len(level_vocab)
                row_ids.append(level_vocab[hierarchy_node])
            rows.append(row_ids)

        return (
            torch.tensor(rows, dtype=torch.long),
            [len(vocab) for vocab in level_vocabularies],
            unmapped,
        )

    def _initial_code_embeddings(self) -> torch.Tensor:
        """Build learnable M by concatenating each code's level embeddings."""
        level_vectors = [
            table(self.code_level_node_ids[:, level])
            for level, table in enumerate(self.hierarchy_embeddings)
        ]
        return torch.cat(level_vectors, dim=-1)

    def _valid_code_mask(self, codes: torch.Tensor) -> torch.Tensor:
        """Mask padding, unknown, and out-of-vocabulary IDs from graph edges."""
        return (
            (codes != self.pad_idx)
            & (codes != self.unk_idx)
            & (codes >= 0)
            & (codes < self.num_nodes)
        )

    def _get_values_and_mask(
        self, value
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unpack a processor value, whose nested-sequence schema may include a mask."""
        if isinstance(value, torch.Tensor):
            return value, None
        schema = self.dataset.input_processors[self.feature_key].schema()
        tensor_values = value[schema.index("value")]
        mask = value[schema.index("mask")] if "mask" in schema else None
        return tensor_values, mask

    def construct_initial_hypergraph(
        self, codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build patient-isolated incidence H and a valid-visit mask.

        H has shape [batch, num_nodes, num_visits]. H[b, i, j] is 1 when input
        diagnosis ID i appears in visit j for patient b. Padding and <unk> IDs
        are excluded because they do not identify a specific diagnosis.
        """
        if codes.dim() != 3:
            raise ValueError(
                f"Expected {self.feature_key} with shape [B,V,C], "
                f"got {tuple(codes.shape)}"
            )
        codes = codes.long()
        batch_size, num_visits, _ = codes.shape
        valid_codes = self._valid_code_mask(codes)
        valid_visits = valid_codes.any(dim=-1)

        # One dense [codes, visits] matrix per patient, rather than a mostly-zero
        # [batch*codes, batch*visits] block-diagonal matrix.
        H = torch.zeros(
            batch_size,
            self.num_nodes,
            num_visits,
            dtype=torch.float32,
            device=codes.device,
        )
        batch_idx, visit_idx, slot_idx = valid_codes.nonzero(as_tuple=True)
        if batch_idx.numel():
            code_ids = codes[batch_idx, visit_idx, slot_idx]
            H[batch_idx, code_ids, visit_idx] = 1.0
        return H, valid_visits

    def message_passing(self, H: torch.Tensor, initial_M: torch.Tensor) -> torch.Tensor:
        """Update code embeddings independently for each patient's hypergraph.

        Args:
            H: Batched incidence tensor [batch, codes, visits].
            initial_M: Shared initial code table [codes, embedding_dim].

        Returns:
            Patient-specific code states [batch, codes, embedding_dim].
        """
        if H.dim() != 3 or initial_M.dim() != 2:
            raise ValueError("Expected H [B,N,V] and initial_M [N,D]")
        if H.size(1) != initial_M.size(0):
            raise ValueError("H code rows must match the rows of initial_M")

        batch_size = H.size(0)
        # Every patient starts from the same learned diagnosis table M. The
        # separate batch axis lets each patient's graph produce its own updates.
        node_states = initial_M.unsqueeze(0).expand(batch_size, -1, -1)

        # A visit's degree is the number of diagnosis nodes connected to it.
        # The clamp keeps empty padded visits finite; their H columns are zero.
        visit_degree = H.sum(dim=1).clamp_min(1.0)  # [B, visits]

        # Repeat the UniGIN update for each layer:
        # 4a. Mean-pool code states into visit states (Hᵀ @ M).
        # 4b. Sum the visit states back to each code (H @ V).
        # 4c. Add the code's own state, transform, and apply LeakyReLU.
        for linear, epsilon in zip(self.message_layers, self.message_eps):
            visit_states = torch.bmm(H.transpose(1, 2), node_states)
            visit_states = visit_states / visit_degree.unsqueeze(-1)
            visit_aggregate = torch.bmm(H, visit_states)
            node_states = self.message_activation(
                linear((1.0 + epsilon) * node_states + visit_aggregate)
            )
        return node_states

    # Top-k turns similarities into integer edge choices, so gradients cannot
    # flow through these scores; skip building an unused autograd graph.
    @torch.no_grad()
    def augment_hypergraph(
        self,
        H: torch.Tensor,
        node_states: torch.Tensor,
        visit_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add high-scoring missing code-to-visit links, patient by patient.

        This follows SHy's missing-diagnosis augmentation: score every currently
        absent code/visit pair with an average of weighted cosine similarities,
        then add ``floor(p * existing_edge_count)`` of the strongest pairs.

        Args:
            H: Original incidence tensor [B, N, V].
            node_states: Updated code embeddings [B, N, D].
            visit_states: Embeddings of the original visits [B, V, D].

        Returns:
            ``(H_augmented, added_edges)``. Both have shape [B, N, V];
            ``added_edges`` contains 1 only for newly selected connections.
        """
        if H.dim() != 3 or node_states.dim() != 3 or visit_states.dim() != 3:
            raise ValueError(
                "Expected H [B,N,V], node_states [B,N,D], visit_states [B,V,D]"
            )
        batch_size, num_nodes, num_visits = H.shape
        if node_states.shape[:2] != (batch_size, num_nodes):
            raise ValueError("node_states batch/code dimensions must match H")
        if visit_states.shape[:2] != (batch_size, num_visits):
            raise ValueError("visit_states batch/visit dimensions must match H")

        added_edges = torch.zeros_like(H)

        # STEP 5a — p=0 is the ablation with missing-diagnosis augmentation off.
        if self.augmentation_ratio == 0 or num_visits == 0 or num_nodes == 0:
            return H, added_edges

        # STEP 5b — Build patient-specific valid row/column masks. The B axis
        # stays intact below; no patient's scores are compared with another's.
        code_is_valid = torch.ones(num_nodes, dtype=torch.bool, device=H.device)
        if 0 <= self.pad_idx < num_nodes:
            code_is_valid[self.pad_idx] = False
        if 0 <= self.unk_idx < num_nodes:
            code_is_valid[self.unk_idx] = False
        visit_is_valid = H.sum(dim=1) > 0  # [B,V], ignoring all-padding visits
        # Each observed edge belongs to a valid visit by construction. Excluding
        # PAD/UNK rows here gives the exact number of absent valid pairs without
        # allocating a second full [B,N,V] Boolean tensor.
        edge_counts = H.sum(dim=(1, 2)).to(torch.long)  # [B]
        valid_edges_per_code = H.sum(dim=2) * code_is_valid[None, :]
        valid_edge_counts = valid_edges_per_code.sum(dim=1).to(torch.long)
        eligible_counts = code_is_valid.sum() * visit_is_valid.sum(dim=1)
        candidate_counts = (eligible_counts - valid_edge_counts).clamp_min(0)
        add_counts = torch.minimum(
            torch.floor(edge_counts.to(node_states.dtype) * self.augmentation_ratio).to(
                torch.long
            ),
            candidate_counts,
        )
        if batch_size == 0:
            return H, added_edges
        max_add_count = int(add_counts.max().item())
        if max_add_count == 0:
            return H, added_edges

        # STEP 5c–d — Score bounded code chunks for the full patient batch, then
        # merge each chunk's top candidates. This avoids both a Python loop over
        # patients and a full [B,N,V] score allocation.
        weighted_visit_heads = [
            F.normalize(
                visit_states * projection[None, None, :],
                p=2,
                dim=-1,
                eps=1e-8,
            )
            for projection in self.similarity_projections
        ]
        best_scores = None
        best_indices = None
        for code_start in range(0, num_nodes, self.phenotype_code_chunk_size):
            code_end = min(code_start + self.phenotype_code_chunk_size, num_nodes)
            code_chunk = node_states[:, code_start:code_end]
            chunk_scores = torch.zeros(
                (batch_size, code_end - code_start, num_visits),
                dtype=node_states.dtype,
                device=H.device,
            )
            for projection, weighted_visits in zip(
                self.similarity_projections, weighted_visit_heads
            ):
                weighted_codes = F.normalize(
                    code_chunk * projection[None, None, :],
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                chunk_scores = chunk_scores + torch.bmm(
                    weighted_codes, weighted_visits.transpose(1, 2)
                )
            chunk_scores = chunk_scores / self.num_similarity_heads

            chunk_is_valid = (
                code_is_valid[code_start:code_end][None, :, None]
                & visit_is_valid[:, None, :]
                & (H[:, code_start:code_end] == 0)
            )
            chunk_scores = chunk_scores.masked_fill(
                ~chunk_is_valid, float("-inf")
            ).reshape(batch_size, -1)
            local_k = min(max_add_count, chunk_scores.size(1))
            local_scores, local_indices = torch.topk(chunk_scores, k=local_k, dim=1)
            # Convert this chunk's flattened [code,visit] offsets to offsets in
            # the patient's full [N,V] matrix.
            local_indices = local_indices + code_start * num_visits

            if best_scores is None:
                best_scores, best_indices = local_scores, local_indices
            else:
                merged_scores = torch.cat([best_scores, local_scores], dim=1)
                merged_indices = torch.cat([best_indices, local_indices], dim=1)
                keep_k = min(max_add_count, merged_scores.size(1))
                best_scores, best_positions = torch.topk(merged_scores, k=keep_k, dim=1)
                best_indices = torch.gather(merged_indices, 1, best_positions)

        # The batch top-k width is the maximum requested across patients; mask
        # out the unused ranks for patients needing fewer added edges.
        rank_is_selected = (
            torch.arange(best_indices.size(1), device=H.device)[None, :]
            < add_counts[:, None]
        )
        added_flat = H.new_zeros((batch_size, num_nodes * num_visits))
        added_flat.scatter_(1, best_indices, rank_is_selected.to(added_flat.dtype))
        added_edges = added_flat.reshape(batch_size, num_nodes, num_visits)

        # STEP 5e — Union the original incidence links with the new links.
        # Addition is safe because the selected pairs were explicitly missing.
        return H + added_edges, added_edges

    def extract_phenotypes(
        self,
        H_augmented: torch.Tensor,
        node_states: torch.Tensor,
        visit_states: torch.Tensor,
        *,
        valid_visits: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score existing diagnosis-visit links and extract K sub-hypergraphs.

        Equation 6 conceptually assigns a mask score to every diagnosis/visit
        pair. Equation 8 then multiplies those masks by H_augmented, so absent
        pairs cannot appear in a phenotype graph or affect distinctness. We can
        therefore score only the links present in H_augmented and avoid building
        the much larger code-by-visit score grid.

        Existing links are processed in code chunks. For each link, its patient's
        diagnosis and visit embeddings are gathered and passed through each
        phenotype MLP. This bounds the intermediate pair features by the number
        of existing links in a chunk, rather than [B,C,V,H]. The full mask tensor
        is not retained; ``mask_gram`` stores Psi_j^T Psi_j for each visit. The
        resulting phenotype sub-hypergraphs are stored sparsely.

        During training, ``sample=True`` (the default) uses a hard-forward,
        straight-through Gumbel-Sigmoid sample, matching the authors' released
        RelaxedBernoulliStraightThrough implementation. In evaluation, masks are
        also sampled by default, as in the authors' code; with
        ``sample_phenotypes_at_eval=False`` (or ``sample=False``) they are
        thresholded deterministically at probability 0.5 instead.

        Returns:
            ``(phenotype_hypergraphs, mask_gram, edge_indices)``:
            sparse COO phenotype graphs [B,K,N,V], per-visit Gamma Gram matrices
            [B,V,K,K], and [E,3] coordinates [patient, code, visit] for the
            augmented edges scored by the phenotype MLPs.
        """
        # Shape guide for the tensors built below:
        #   H_augmented[b, code, visit]       -> one patient's augmented graph
        #   edge_mask_chunks[e, k]             -> edge e's mask for phenotype k
        #   mask_gram[b, visit, k, l]         -> overlap of two phenotype masks
        #   edge indices                      -> [patient, code, visit]
        #   sparse tensor coordinates         -> [patient, phenotype, code, visit]
        if H_augmented.dim() != 3 or node_states.dim() != 3 or visit_states.dim() != 3:
            raise ValueError(
                "Expected H_augmented [B,N,V], node_states [B,N,D], "
                "and visit_states [B,V,D]"
            )
        batch_size, num_nodes, num_visits = H_augmented.shape
        if node_states.shape[:2] != (batch_size, num_nodes):
            raise ValueError("node_states batch/code dimensions must match H_augmented")
        if visit_states.shape[:2] != (batch_size, num_visits):
            raise ValueError(
                "visit_states batch/visit dimensions must match H_augmented"
            )
        if node_states.size(-1) != visit_states.size(-1):
            raise ValueError("Diagnosis and visit embedding dimensions must match")

        # If callers do not pass the mask from H construction, infer valid visits
        # from nonempty columns. Padded visits must not contribute to Gamma or loss.
        if valid_visits is None:
            valid_visits = H_augmented.sum(dim=1) > 0
        else:
            valid_visits = valid_visits.to(device=H_augmented.device, dtype=torch.bool)
        if tuple(valid_visits.shape) != (batch_size, num_visits):
            raise ValueError("valid_visits must have shape [B,V]")

        if sample is None:
            sample = self.training or getattr(self, "sample_phenotypes_at_eval", True)

        # valid_code_rows has shape [N]: one Boolean per code row. PAD and UNK
        # are special rows, so they are marked False instead of being diagnoses.
        valid_code_rows = torch.ones(
            num_nodes, dtype=torch.bool, device=H_augmented.device
        )
        if 0 <= self.pad_idx < num_nodes:
            valid_code_rows[self.pad_idx] = False
        if 0 <= self.unk_idx < num_nodes:
            valid_code_rows[self.unk_idx] = False

        # STEP 6a — Find augmented edges once. These are the only coordinates
        # that can survive the final multiplication by H_augmented (Eq. 8).
        edge_indices = (H_augmented > 0).nonzero(
            as_tuple=False
        )  # [E,3]: b, code, visit
        # Keep code IDs for chunk selection; full coordinates are gathered later.
        edge_codes = edge_indices[:, 1]

        # STEP 6b — Score only augmented links, in code chunks.
        # Each chosen row remains [patient, code, visit], so gathered diagnosis
        # and visit embeddings always come from the same patient's history.
        mask_gram = node_states.new_zeros(
            (batch_size, num_visits, self.num_phenotypes, self.num_phenotypes)
        )
        edge_mask_chunks = []
        edge_coordinate_chunks = []
        for code_start in range(0, num_nodes, self.phenotype_code_chunk_size):
            code_end = min(code_start + self.phenotype_code_chunk_size, num_nodes)
            in_chunk = (edge_codes >= code_start) & (edge_codes < code_end)
            chosen_coordinates = edge_indices[in_chunk]
            if chosen_coordinates.size(0) == 0:
                continue

            edge_patients, edge_codes_in_chunk, edge_visits = chosen_coordinates.unbind(
                dim=1
            )
            edge_code_states = node_states[edge_patients, edge_codes_in_chunk]
            edge_visit_states = visit_states[edge_patients, edge_visits]

            # Evaluate the MLP only on real [patient, code, visit] links. Split
            # the first Linear's weight across the code and visit halves so its
            # result matches Linear([code_embedding || visit_embedding]) without
            # constructing a dense [B,C,V,2D] pair-feature table.
            logits_by_phenotype = []
            for extractor in self.phenotype_extractors:
                first_linear, activation, output_linear = extractor
                code_hidden = F.linear(
                    edge_code_states,
                    first_linear.weight[:, : node_states.size(-1)],
                )  # [E_chunk,H], no bias yet
                visit_hidden = F.linear(
                    edge_visit_states,
                    first_linear.weight[:, node_states.size(-1) :],
                    first_linear.bias,
                )  # [E_chunk,H], includes the first-layer bias once
                pair_hidden = activation(code_hidden + visit_hidden)
                logits_by_phenotype.append(output_linear(pair_hidden).squeeze(-1))
            chunk_logits = torch.stack(logits_by_phenotype, dim=1)  # [E_chunk,K]

            # STEP 6c — Sample or threshold masks for these existing links.
            if sample:
                uniform = torch.rand_like(chunk_logits).clamp_(1e-6, 1.0 - 1e-6)
                logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
                relaxed = torch.sigmoid(
                    (chunk_logits + logistic_noise) / self.phenotype_temperature
                )
                hard_mask = (relaxed >= 0.5).to(relaxed.dtype)
                chunk_masks = hard_mask - relaxed.detach() + relaxed
            else:
                chunk_masks = (chunk_logits >= 0).to(chunk_logits.dtype)

            # PAD/UNK rows and padded visit columns are not valid phenotype links.
            valid_edges = (
                valid_code_rows[edge_codes_in_chunk]
                & valid_visits[edge_patients, edge_visits]
            )
            selected_masks = chunk_masks * valid_edges[:, None].to(chunk_masks.dtype)
            edge_coordinate_chunks.append(chosen_coordinates)
            edge_mask_chunks.append(selected_masks)

        # Assemble the sparse COO phenotype sub-hypergraphs from the saved edge
        # coordinates and their K mask values. No dense [B,K,N,V] copy is made.
        if edge_mask_chunks:
            # Concatenate chunk rows: [E1,3] + [E2,3] -> [E,3].
            subgraph_edge_indices = torch.cat(edge_coordinate_chunks, dim=0)
            # Matching mask rows: [E1,K] + [E2,K] -> [E,K].
            edge_mask_values = torch.cat(edge_mask_chunks, dim=0)  # [E,K]
        else:
            subgraph_edge_indices = torch.empty(
                (0, 3), dtype=torch.long, device=H_augmented.device
            )
            edge_mask_values = node_states.new_empty((0, self.num_phenotypes))
        subgraph_patients, subgraph_codes, subgraph_visits = (
            subgraph_edge_indices.unbind(dim=1)
        )

        # Per-visit Gram matrices [B,V,K,K] for L_distinct. As in the authors'
        # code (enriched_H *= incident_mask before the distinctness loss), only
        # links in H_augmented count, so codes the patient does not have cannot
        # dominate the loss. Each link adds the outer product of its K mask values.
        if edge_mask_values.numel() > 0:
            flat_visit_ids = subgraph_patients * num_visits + subgraph_visits
            link_outer = edge_mask_values.unsqueeze(2) * edge_mask_values.unsqueeze(1)
            mask_gram = (
                mask_gram.reshape(-1, self.num_phenotypes, self.num_phenotypes)
                .index_add(0, flat_visit_ids, link_outer)
                .reshape_as(mask_gram)
            )
        num_subgraph_edges = subgraph_edge_indices.size(0)
        # phenotype_ids is [K,E]: each column is one link; rows list phenotype
        # IDs 0..K-1 so every link gets one sparse coordinate per phenotype.
        phenotype_ids = (
            torch.arange(self.num_phenotypes, device=H_augmented.device)
            .unsqueeze(1)
            .expand(-1, num_subgraph_edges)
        )
        # Build sparse coordinate rows in matching K-major order:
        # [patient, phenotype, code, visit], one column per stored value.
        sparse_indices = torch.stack(
            [
                subgraph_patients.unsqueeze(0)
                .expand(self.num_phenotypes, -1)
                .reshape(-1),
                phenotype_ids.reshape(-1),
                subgraph_codes.unsqueeze(0).expand(self.num_phenotypes, -1).reshape(-1),
                subgraph_visits.unsqueeze(0)
                .expand(self.num_phenotypes, -1)
                .reshape(-1),
            ],
            dim=0,
        )
        # [E,K] -> [K,E] -> flat, matching phenotype-major sparse_indices order.
        sparse_values = edge_mask_values.transpose(0, 1).reshape(-1)
        phenotype_hypergraphs = torch.sparse_coo_tensor(
            sparse_indices,
            sparse_values,
            size=(batch_size, self.num_phenotypes, num_nodes, num_visits),
            device=H_augmented.device,
        ).coalesce()
        return phenotype_hypergraphs, mask_gram, subgraph_edge_indices

    def embed_phenotypes_over_time(
        self,
        phenotype_hypergraphs: torch.Tensor,
        node_states: torch.Tensor,
        *,
        valid_visits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed each phenotype's diagnosis links over its visit sequence.

        This implements the paper's Equation 9 and the following GRU/attention
        steps. For every patient and phenotype, selected diagnosis embeddings are
        summed into one vector per visit, passed through a shared GRU, then
        combined with attention over valid visits.

        Args:
            phenotype_hypergraphs: Sparse COO ``[B,K,N,V]`` tensor produced by
                :meth:`extract_phenotypes` (the nonzero values are mask weights).
            node_states: Patient-specific diagnosis embeddings ``[B,N,D]``.
            valid_visits: Optional Boolean visit mask ``[B,V]`` from the original
                patient hypergraph. Pass this to keep visits valid even if a
                phenotype selected no diagnoses there. If omitted, visits are
                inferred from the union of edges across phenotype graphs.

        Returns:
            ``(phenotype_vectors, attention_weights, phenotype_visit_embeddings)``:
            vectors ``[B,K,H]``, visit weights ``[B,K,V]``, and the summed
            pre-GRU visit embeddings ``[B,K,V,D]``.
        """
        if phenotype_hypergraphs.layout != torch.sparse_coo:
            raise ValueError("phenotype_hypergraphs must use sparse COO layout")
        if phenotype_hypergraphs.dim() != 4 or node_states.dim() != 3:
            raise ValueError(
                "Expected phenotype graphs [B,K,N,V] and node_states [B,N,D]"
            )

        batch_size, num_phenotypes, num_nodes, num_visits = phenotype_hypergraphs.shape
        if node_states.shape[:2] != (batch_size, num_nodes):
            raise ValueError(
                "node_states batch/code dimensions must match phenotype graphs"
            )
        if node_states.size(-1) != self.phenotype_gru.input_size:
            raise ValueError("node_states embedding width must match the phenotype GRU")

        # Sparse edge coordinates are [patient, phenotype, code, visit]. Each
        # coordinate contributes its mask value times that patient's code vector.
        coalesced_graphs = phenotype_hypergraphs.coalesce()
        edge_coordinates = coalesced_graphs.indices()
        edge_values = coalesced_graphs.values().to(dtype=node_states.dtype)
        visit_sums = node_states.new_zeros(
            (batch_size * num_phenotypes * num_visits, node_states.size(-1))
        )
        if edge_values.numel() > 0 and num_visits > 0:
            edge_patients, edge_phenotypes, edge_codes, edge_visits = (
                edge_coordinates.unbind(dim=0)
            )
            flat_visit_ids = (
                edge_patients * num_phenotypes + edge_phenotypes
            ) * num_visits + edge_visits
            selected_code_states = node_states[edge_patients, edge_codes]
            weighted_code_states = selected_code_states * edge_values.unsqueeze(-1)
            visit_sums = visit_sums.index_add(0, flat_visit_ids, weighted_code_states)

        # Restore axes: [patient, phenotype, visit, embedding]. This is ΨᵀM:
        # a sum of the diagnosis embeddings selected for each phenotype/visit.
        phenotype_visit_embeddings = visit_sums.reshape(
            batch_size,
            num_phenotypes,
            num_visits,
            node_states.size(-1),
        )

        if valid_visits is None:
            # Without the original H mask, only visits represented by at least
            # one stored phenotype edge can be identified as valid.
            valid_visits = torch.zeros(
                (batch_size, num_visits), dtype=torch.bool, device=node_states.device
            )
            active_edge_mask = edge_values != 0
            if active_edge_mask.any():
                active_coordinates = edge_coordinates[:, active_edge_mask]
                valid_visits[active_coordinates[0], active_coordinates[3]] = True
        else:
            valid_visits = valid_visits.to(device=node_states.device, dtype=torch.bool)
        if tuple(valid_visits.shape) != (batch_size, num_visits):
            raise ValueError("valid_visits must have shape [B,V]")

        # GRU treats each (patient, phenotype) pair as an independent sequence.
        flattened_sequences = phenotype_visit_embeddings.reshape(
            batch_size * num_phenotypes, num_visits, node_states.size(-1)
        )
        flattened_visit_mask = (
            valid_visits[:, None, :]
            .expand(batch_size, num_phenotypes, num_visits)
            .reshape(batch_size * num_phenotypes, num_visits)
        )

        if num_visits == 0:
            visit_hidden_states = node_states.new_zeros(
                (batch_size * num_phenotypes, 0, self.hidden_dim)
            )
            flat_attention_weights = node_states.new_zeros(
                (batch_size * num_phenotypes, 0)
            )
        else:
            visit_hidden_states, _ = self.phenotype_gru(flattened_sequences)
            attention_logits = self.temporal_attention(visit_hidden_states).squeeze(-1)

            # Mask padded visits before softmax. For a patient with no valid
            # visits, temporarily use zeros to avoid softmax([-inf,...])=NaN,
            # then explicitly return all-zero weights and vector.
            has_valid_visit = flattened_visit_mask.any(dim=1, keepdim=True)
            attention_logits = attention_logits.masked_fill(
                ~flattened_visit_mask, float("-inf")
            )
            attention_logits = torch.where(
                has_valid_visit, attention_logits, torch.zeros_like(attention_logits)
            )
            flat_attention_weights = torch.softmax(attention_logits, dim=-1)
            flat_attention_weights = flat_attention_weights * flattened_visit_mask.to(
                flat_attention_weights.dtype
            )
            flat_attention_weights = (
                flat_attention_weights
                / flat_attention_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            )

        # Attention weights sum the GRU outputs across time into one vector per
        # phenotype. Invalid visits get zero weight; empty histories yield zero.
        flat_vectors = torch.sum(
            visit_hidden_states * flat_attention_weights.unsqueeze(-1), dim=1
        )
        phenotype_vectors = flat_vectors.reshape(
            batch_size, num_phenotypes, self.hidden_dim
        )
        attention_weights = flat_attention_weights.reshape(
            batch_size, num_phenotypes, num_visits
        )
        return phenotype_vectors, attention_weights, phenotype_visit_embeddings

    def predict_from_phenotypes(
        self, phenotype_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply SHy's phenotype self-attention and per-phenotype classifier.

        This implements Equations 13–15. Self-attention lets phenotype vectors
        exchange information, Equation 14 produces one importance weight per
        phenotype, and Equation 15 mixes separate diagnosis predictions from
        the original phenotype vectors.

        Args:
            phenotype_vectors: ``[B,K,H]`` vectors from Equation 12.

        Returns:
            ``(prediction_probs, phenotype_weights, phenotype_probs)`` with
            shapes ``[B,C]``, ``[B,K]``, and ``[B,K,C]`` respectively.
        """
        if phenotype_vectors.dim() != 3:
            raise ValueError("phenotype_vectors must have shape [B,K,H]")
        if phenotype_vectors.size(-1) != self.hidden_dim:
            raise ValueError("phenotype vector width must match hidden_dim")

        # Equation 13: project U into query/key/value features, then run
        # scaled dot-product multi-head attention across the phenotype axis.
        queries = self.phenotype_query_projection(phenotype_vectors)
        keys = self.phenotype_key_projection(phenotype_vectors)
        values = self.phenotype_value_projection(phenotype_vectors)
        attended_phenotypes, _ = self.phenotype_self_attention(
            queries,
            keys,
            values,
            need_weights=False,
        )

        # Equation 14: one score per phenotype, normalized across K.
        importance_logits = self.phenotype_importance(attended_phenotypes).squeeze(-1)
        phenotype_weights = torch.softmax(importance_logits, dim=1)

        # Equation 15: each original U_k gets its own diagnosis distribution;
        # alpha then combines those distributions (rather than combining U first).
        phenotype_logits = self.diagnosis_classifier(phenotype_vectors)
        if self.prediction_activation == "softmax":
            phenotype_probs = torch.softmax(phenotype_logits, dim=-1)
        else:
            phenotype_probs = torch.sigmoid(phenotype_logits)
        prediction_probs = torch.sum(
            phenotype_weights.unsqueeze(-1) * phenotype_probs,
            dim=1,
        )
        return prediction_probs, phenotype_weights, phenotype_probs

    def reconstruct_hypergraph(
        self,
        phenotype_vectors: torch.Tensor,
        original_H: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the original patient hypergraphs from their phenotype vectors.

        Args:
            phenotype_vectors: ``[B,K,H]`` Equation 12 vectors.
            original_H: Original incidence tensor ``[B,N,V]``. The decoder uses
                its shape and values as the reconstruction target.

        Returns:
            Reconstruction logits with shape ``[B,N,V]``. Sigmoid converts
            these to per-code/per-visit probabilities for fidelity BCE.
        """
        batch_size, num_nodes, num_visits = original_H.shape
        if phenotype_vectors.shape[:2] != (batch_size, self.num_phenotypes):
            raise ValueError("phenotype_vectors batch/phenotype axes do not match SHy")
        if num_nodes != self.num_nodes:
            raise ValueError("original_H must match the input-code vocabulary")

        # Concatenate U_1..U_K per patient into one [K * hidden_dim] context.
        context = phenotype_vectors.reshape(batch_size, -1)
        if num_visits == 0:
            return original_H.new_empty((batch_size, num_nodes, 0))
        # Appendix C feeds the same concatenated context to every GRU time step.
        # Shape [B,T,K*H]: each row's context is repeated across its T visits.
        decoder_sequence = context.unsqueeze(1).expand(-1, num_visits, -1)
        decoded_states, _ = self.hypergraph_decoder_gru(decoder_sequence)
        # GRU/output are [B,T,N]; transpose to incidence convention [B,N,T].
        return self.hypergraph_decoder_output(decoded_states).transpose(1, 2)

    def compute_paper_losses(
        self,
        prediction_probs: torch.Tensor,
        y_true: torch.Tensor,
        reconstruction_logits: torch.Tensor,
        original_H: torch.Tensor,
        valid_visits: torch.Tensor,
        mask_gram: torch.Tensor,
        phenotype_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculate SHy's prediction, fidelity, distinctness, and alpha losses.

        The reductions follow the paper: BCE for diagnosis prediction and graph
        reconstruction; per-patient averaging over valid visits for fidelity
        and distinctness; and per-patient averaging for the alpha regularizer.
        The final scalar is Eq. 16 with configurable epsilon/eta/omega weights.
        """
        y_true = y_true.to(device=prediction_probs.device, dtype=prediction_probs.dtype)
        if y_true.shape != prediction_probs.shape:
            raise ValueError(
                f"next-visit labels must have shape {tuple(prediction_probs.shape)}, "
                f"got {tuple(y_true.shape)}"
            )
        prediction_eps = 1e-7
        loss_pred = F.binary_cross_entropy(
            prediction_probs.clamp(prediction_eps, 1.0 - prediction_eps), y_true
        )

        # Fidelity: reconstruct each original diagnosis-by-visit matrix. Ignore
        # padded visit columns; average each patient's BCE over its real visits
        # and code vocabulary, then average patients (including empty histories
        # as zero contribution rather than producing NaN).
        if reconstruction_logits.shape != original_H.shape:
            raise ValueError(
                "reconstruction logits must have the same [B,N,V] shape as H"
            )
        valid_visits = valid_visits.to(device=original_H.device, dtype=torch.bool)
        reconstruction_targets = original_H.to(
            device=reconstruction_logits.device, dtype=reconstruction_logits.dtype
        )
        fidelity_cell_loss = F.binary_cross_entropy_with_logits(
            reconstruction_logits, reconstruction_targets, reduction="none"
        )
        fidelity_mask = valid_visits[:, None, :].to(fidelity_cell_loss.dtype)
        fidelity_per_patient = (fidelity_cell_loss * fidelity_mask).sum(dim=(1, 2))
        fidelity_denominator = (
            valid_visits.sum(dim=1).to(fidelity_cell_loss.dtype) * original_H.size(1)
        ).clamp_min(1.0)
        loss_fidelity = (fidelity_per_patient / fidelity_denominator).mean()

        # Distinctness: mask_gram[b,v] is Gamma_v^T Gamma_v, shape [K,K].
        # Penalize its Frobenius distance from identity for real visits only
        # (the authors' torch.norm(..., p=2) on a matrix is the Frobenius norm).
        num_phenotypes = mask_gram.size(-1)
        identity = torch.eye(
            num_phenotypes, device=mask_gram.device, dtype=mask_gram.dtype
        )
        gram_error = identity - mask_gram
        distinct_per_visit = torch.linalg.matrix_norm(
            gram_error, ord="fro", dim=(-2, -1)
        )  # [B,V], matrix Frobenius norm per patient and visit
        distinct_mask = valid_visits.to(
            device=distinct_per_visit.device, dtype=distinct_per_visit.dtype
        )
        distinct_per_patient = (distinct_per_visit * distinct_mask).sum(dim=1)
        loss_distinct = (
            distinct_per_patient / distinct_mask.sum(dim=1).clamp_min(1.0)
        ).mean()

        # Alpha regularizer from the paper: balance deviation from uniformity
        # against the L2 magnitude, computed independently for each patient.
        alpha_mean = phenotype_weights.mean(dim=1, keepdim=True)
        # This is sqrt(sum((alpha - mean)^2) / K), written as a vector norm
        # divided by sqrt(K). vector_norm has a stable zero-gradient at a
        # perfectly uniform alpha row, unlike sqrt(0) in the literal form.
        alpha_spread = torch.linalg.vector_norm(
            phenotype_weights - alpha_mean, ord=2, dim=1
        ) / (phenotype_weights.size(1) ** 0.5)
        alpha_norm = torch.linalg.vector_norm(phenotype_weights, ord=2, dim=1)
        loss_alpha = -(alpha_spread - alpha_norm).mean()

        loss_total = (
            loss_pred
            + self.fidelity_loss_weight * loss_fidelity
            + self.distinct_loss_weight * loss_distinct
            + self.alpha_loss_weight * loss_alpha
        )
        return {
            "loss": loss_total,
            "loss_pred": loss_pred,
            "loss_fidelity": loss_fidelity,
            "loss_distinct": loss_distinct,
            "loss_alpha": loss_alpha,
        }

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # STEP 1 — Decode the processed batch fields needed by this model.
        # The integer IDs in `codes` use exactly the input processor mapping used
        # above to create code_level_node_ids and therefore the rows of M.
        codes, _processor_mask = self._get_values_and_mask(kwargs[self.feature_key])
        codes = codes.to(self.device).long()

        # STEP 2 — Build each patient's incidence matrix H from history codes.
        H, _valid_visits = self.construct_initial_hypergraph(codes)

        # STEP 3 — Assemble M from the learnable per-level hierarchy embeddings.
        # M is [num_nodes, embedding_dim].
        initial_M = self._initial_code_embeddings()

        # STEP 4 — Apply stacked UniGIN-style code -> visit -> code updates.
        node_states = self.message_passing(H, initial_M)

        # STEP 5 — Build visit embeddings from the ORIGINAL patient hypergraphs.
        # SHy's augmentation scores compare updated code states with these visit
        # states, then H_aug records the extra likely-missing diagnoses.
        visit_degree = H.sum(dim=1).clamp_min(1.0)
        visit_states = torch.bmm(H.transpose(1, 2), node_states)
        visit_states = visit_states / visit_degree.unsqueeze(-1)
        H_augmented, _added_edges = self.augment_hypergraph(
            H, node_states, visit_states
        )

        # STEP 6 — Extract the K phenotype-specific sub-hypergraphs.
        # These are sparse Ψ tensors, one masked graph per patient and phenotype.
        phenotype_hypergraphs, phenotype_mask_gram, phenotype_edge_indices = (
            self.extract_phenotypes(
                H_augmented,
                node_states,
                visit_states,
                valid_visits=_valid_visits,
            )
        )

        # STEP 7 — Encode each extracted phenotype through time (Eq. 9 onward).
        phenotype_vectors, temporal_attention_weights, phenotype_visit_embeddings = (
            self.embed_phenotypes_over_time(
                phenotype_hypergraphs,
                node_states,
                valid_visits=_valid_visits,
            )
        )

        # STEP 8–9 — Weight phenotypes and combine their separate predictions.
        prediction_probs, phenotype_weights, phenotype_prediction_probs = (
            self.predict_from_phenotypes(phenotype_vectors)
        )

        # STEP 10 — Decode the original history for the fidelity objective.
        reconstruction_logits = self.reconstruct_hypergraph(phenotype_vectors, H)

        # PyHealth's default multilabel loss expects logits. Convert the final
        # mixture probabilities back to equivalent logits so sigmoid(logit)
        # recovers exactly the selected softmax/sigmoid mixture probabilities.
        probability_eps = 1e-7
        if self.mode == "multiclass":
            logits = prediction_probs.clamp_min(probability_eps).log()
        else:
            logits = torch.logit(
                prediction_probs.clamp(min=probability_eps, max=1.0 - probability_eps)
            )

        result = {"logit": logits, "y_prob": self.prepare_y_prob(logits)}
        # Expose intermediate phenotype structures and temporal vectors on request.
        if kwargs.get("return_phenotypes", False):
            result["phenotype_hypergraphs"] = phenotype_hypergraphs
            result["phenotype_mask_gram"] = phenotype_mask_gram
            result["phenotype_edge_indices"] = phenotype_edge_indices
            result["phenotype_visit_embeddings"] = phenotype_visit_embeddings
            result["phenotype_vectors"] = phenotype_vectors
            result["temporal_attention_weights"] = temporal_attention_weights
            result["phenotype_importance_weights"] = phenotype_weights
            result["phenotype_prediction_probs"] = phenotype_prediction_probs
            result["prediction_probs"] = prediction_probs
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            result["y_true"] = y_true
            # Compute the paper's full objective, rather than only PyHealth's
            # prediction loss. The current task's labels are multi-hot, so the
            # prediction term is the paper's elementwise binary cross-entropy.
            result.update(
                self.compute_paper_losses(
                    prediction_probs,
                    y_true,
                    reconstruction_logits,
                    H,
                    _valid_visits,
                    phenotype_mask_gram,
                    phenotype_weights,
                )
            )
        if kwargs.get("return_phenotypes", False):
            result["reconstruction_probs"] = torch.sigmoid(reconstruction_logits)
        if kwargs.get("embed", False):
            # Keep the phenotype axis: SHy predicts from K separate concepts.
            result["embed"] = phenotype_vectors
        return result
