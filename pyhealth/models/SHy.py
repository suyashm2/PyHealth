"""A PyHealth 2.1 compatible SHy starting point for next-visit diagnosis prediction.

The model builds hierarchy-aware initial code embeddings, applies a batched
UniGIN-style code -> visit -> code message-passing step on patient-isolated
hypergraphs, augments likely missing diagnosis-visit links, extracts
phenotype-specific sub-hypergraphs, and embeds each phenotype over time with a
GRU and visit attention. Phenotype-level prediction and auxiliary explanation
objectives are still future work.

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


class SHy(BaseModel):
    """Predict multi-label next-visit diagnoses from nested diagnosis histories."""

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
        phenotype_code_chunk_size: int = 128,
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
        if num_similarity_heads <= 0:
            raise ValueError("num_similarity_heads must be positive")
        if num_phenotypes <= 0:
            raise ValueError("num_phenotypes must be positive")
        if phenotype_temperature <= 0:
            raise ValueError("phenotype_temperature must be positive")
        if phenotype_code_chunk_size <= 0:
            raise ValueError("phenotype_code_chunk_size must be positive")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]
        self.embedding_dim = embedding_dim
        self.num_hierarchy_levels = num_hierarchy_levels
        self.per_level_dim = embedding_dim // num_hierarchy_levels
        self.augmentation_ratio = float(augmentation_ratio)
        self.num_similarity_heads = num_similarity_heads
        self.num_phenotypes = num_phenotypes
        self.phenotype_temperature = float(phenotype_temperature)
        self.phenotype_code_chunk_size = phenotype_code_chunk_size
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
        self.num_nodes = input_processor.size()
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
            nn.init.normal_(table.weight, mean=0.0, std=0.02)

        # ---------------------------------------------------------------------
        # STEP 4 — Set up patient-specific UniGIN-style message passing.
        # 4a. H chooses which code rows contribute to each visit.
        # 4b. Each layer averages code vectors into visits, then sums those visit
        #     vectors back into codes while retaining each code's own state.
        # 4c. Parameters are shared across patients; batched H keeps their data
        #     and updated code states separate.
        # ---------------------------------------------------------------------
        self.message_layers = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in range(num_message_passing_layers)]
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

        # Phenotype-level prediction is not implemented yet. This pooling and
        # classifier head remains a temporary downstream prediction scaffold.
        self.patient_projection = nn.Linear(embedding_dim, hidden_dim)
        self.patient_activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, self.get_output_size())

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

    def _get_values_and_mask(self, value) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
                f"Expected {self.feature_key} with shape [B,V,C], got {tuple(codes.shape)}"
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

    def message_passing(
        self, H: torch.Tensor, initial_M: torch.Tensor
    ) -> torch.Tensor:
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
            raise ValueError("Expected H [B,N,V], node_states [B,N,D], visit_states [B,V,D]")
        batch_size, num_nodes, num_visits = H.shape
        if node_states.shape[:2] != (batch_size, num_nodes):
            raise ValueError("node_states batch/code dimensions must match H")
        if visit_states.shape[:2] != (batch_size, num_visits):
            raise ValueError("visit_states batch/visit dimensions must match H")

        added_edges = torch.zeros_like(H)

        # STEP 5a — p=0 is the ablation with missing-diagnosis augmentation off.
        if self.augmentation_ratio == 0 or num_visits == 0 or num_nodes == 0:
            return H, added_edges

        # Process one patient at a time. Scores are only [codes, visits], so
        # memory stays patient-local and no patient's visits can affect another.
        for patient_idx in range(batch_size):
            patient_H = H[patient_idx]
            existing_edge_count = int(patient_H.sum().item())
            if existing_edge_count == 0:
                # With no observed visits there is no evidence to augment from.
                continue

            # STEP 5b — Determine eligible rows and columns.
            # Padding/unknown code rows and all-padding visit columns are never
            # candidates. Existing connections are masked in the scoring step.
            code_is_valid = torch.ones(num_nodes, dtype=torch.bool, device=H.device)
            if 0 <= self.pad_idx < num_nodes:
                code_is_valid[self.pad_idx] = False
            if 0 <= self.unk_idx < num_nodes:
                code_is_valid[self.unk_idx] = False
            visit_is_valid = patient_H.sum(dim=0) > 0
            if not code_is_valid.any() or not visit_is_valid.any():
                continue

            # STEP 5c — Compute the paper's multi-head weighted-cosine scores.
            # Each head applies its learned feature weights before normalizing;
            # the dot product is then cosine similarity for every code/visit pair.
            scores = torch.zeros(
                num_nodes, num_visits, dtype=node_states.dtype, device=H.device
            )
            for projection in self.similarity_projections:
                weighted_codes = F.normalize(
                    node_states[patient_idx] * projection.unsqueeze(0),
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                weighted_visits = F.normalize(
                    visit_states[patient_idx] * projection.unsqueeze(0),
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                scores = scores + weighted_codes @ weighted_visits.transpose(0, 1)
            scores = scores / self.num_similarity_heads

            # A pair is selectable only if its code row and visit column are
            # valid AND it is not already an edge in the original hypergraph.
            eligible_pairs = code_is_valid.unsqueeze(1) & visit_is_valid.unsqueeze(0)
            missing_pairs = (patient_H == 0) & eligible_pairs
            scores = scores.masked_fill(~missing_pairs, float("-inf"))

            # STEP 5d — Add the strongest k missing links for this patient.
            # k uses the number of original edges (the paper's p hyperparameter),
            # capped by available candidates so top-k remains well-defined.
            num_candidates = int(missing_pairs.sum().item())
            num_to_add = min(
                int(self.augmentation_ratio * existing_edge_count), num_candidates
            )
            if num_to_add == 0:
                continue
            selected = torch.topk(scores.flatten(), k=num_to_add).indices
            selected_codes = torch.div(selected, num_visits, rounding_mode="floor")
            selected_visits = selected.remainder(num_visits)
            added_edges[patient_idx, selected_codes, selected_visits] = 1.0

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
        """Score all diagnosis-visit pairs and extract K sub-hypergraphs.

        Equation 6 defines mask scores for every diagnosis/visit pair, including
        pairs absent from H_augmented. Those masks are needed by SHy's distinctness
        objective. Equation 8 then multiplies each mask by H_augmented, so only
        existing augmented edges appear in the phenotype sub-hypergraphs.

        Pair embeddings are evaluated in code-row chunks to avoid constructing a
        huge [B,N,V,2D] tensor. The full mask tensor is not retained; instead,
        ``mask_gram`` stores Gamma_j^T Gamma_j for every visit, which is the exact
        sufficient statistic used by the paper's distinctness objective. The
        resulting phenotype sub-hypergraphs are stored sparsely.

        During training, ``sample=True`` (the default) uses a hard-forward,
        straight-through Gumbel-Sigmoid sample, matching the authors' released
        RelaxedBernoulliStraightThrough implementation. In evaluation, the default
        is deterministic thresholding at probability 0.5.

        Returns:
            ``(phenotype_hypergraphs, mask_gram, edge_indices)``:
            sparse COO phenotype graphs [B,K,N,V], per-visit Gamma Gram matrices
            [B,V,K,K], and [E,3] coordinates [patient, code, visit] for the
            augmented edges stored in the sparse phenotype graphs. The MLPs score
            all valid code/visit pairs, not just these E stored graph edges.
        """
        # Shape guide for the tensors built below:
        #   H_augmented[b, code, visit]       -> one patient's augmented graph
        #   chunk_masks[k, local_code, visit] -> one phenotype's mask values
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
            raise ValueError("visit_states batch/visit dimensions must match H_augmented")
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
            sample = self.training

        # valid_code_rows has shape [N]: one Boolean per code row. PAD and UNK
        # are special rows, so they are marked False instead of being diagnoses.
        valid_code_rows = torch.ones(num_nodes, dtype=torch.bool, device=H_augmented.device)
        if 0 <= self.pad_idx < num_nodes:
            valid_code_rows[self.pad_idx] = False
        if 0 <= self.unk_idx < num_nodes:
            valid_code_rows[self.unk_idx] = False

        # STEP 6a — Find augmented edges once. These are the only coordinates
        # that can survive the final multiplication by H_augmented (Eq. 8).
        edge_indices = (H_augmented > 0).nonzero(as_tuple=False)  # [E,3]: b, code, visit
        # Unbind the three coordinate columns into E-length vectors.
        edge_patients, edge_codes, edge_visits = edge_indices.unbind(dim=1)

        # STEP 6b — Score the complete code x visit grid in manageable chunks.
        # Each patient has its own node/visit embeddings. For each code chunk,
        # build [codes-in-chunk, visits, 2D] pair features and score every pair.
        all_patient_grams = []
        edge_mask_chunks = []
        edge_coordinate_chunks = []
        for patient_idx in range(batch_size):
            # Sum Gamma^T Gamma by visit as chunks are processed. This retains the
            # full information needed by L_distinct without storing dense Gamma.
            # Rows/columns of patient_gram are phenotype IDs (0..K-1).
            # Its leading visit axis picks the visit whose diagnosis masks overlap.
            patient_gram = node_states.new_zeros(
                (num_visits, self.num_phenotypes, self.num_phenotypes)
            )

            for code_start in range(0, num_nodes, self.phenotype_code_chunk_size):
                code_end = min(code_start + self.phenotype_code_chunk_size, num_nodes)
                code_chunk = node_states[patient_idx, code_start:code_end]  # rows: C codes; cols: D features
                code_count = code_chunk.size(0)

                # Broadcast each code in this chunk against every visit, then
                # concatenate embeddings as Equation 6 specifies: [M_i || V_j].
                code_pairs = code_chunk[:, None, :].expand(
                    code_count, num_visits, -1
                )  # rows: code-in-chunk; columns: visit; last axis: D features
                visit_pairs = visit_states[patient_idx][None, :, :].expand(
                    code_count, -1, -1
                )  # same [C,V,D] grid, with each visit repeated for each code
                pair_features = torch.cat(
                    [code_pairs, visit_pairs], dim=-1
                )  # [C,V,2D]: each cell is [code embedding || visit embedding]

                # Each phenotype's MLP scores every pair in this chunk.
                chunk_logits = torch.stack(
                    [
                        extractor(pair_features).squeeze(-1)
                        for extractor in self.phenotype_extractors
                    ],
                    dim=0,
                )  # [K,C,V]: axes are phenotype, code-in-chunk, visit

                # STEP 6c — Apply Gumbel-Sigmoid to all pairs, not only H~ edges.
                if sample:
                    uniform = torch.rand_like(chunk_logits).clamp_(1e-6, 1.0 - 1e-6)
                    logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
                    relaxed = torch.sigmoid(
                        (chunk_logits + logistic_noise) / self.phenotype_temperature
                    )
                    # Hard forward decisions (0/1), relaxed gradients backward.
                    # This is the straight-through behavior used by the authors.
                    hard_mask = (relaxed >= 0.5).to(relaxed.dtype)
                    chunk_masks = hard_mask - relaxed.detach() + relaxed
                else:
                    # Deterministic test-time mask, equivalent to sigmoid(logit)>=0.5.
                    chunk_masks = (chunk_logits >= 0).to(chunk_logits.dtype)

                # PAD/UNK rows and padded visits do not represent real pairs.
                chunk_code_valid = valid_code_rows[code_start:code_end]
                # chunk_code_valid is [C]; valid_visits[patient_idx] is [V].
                # Add singleton axes so AND broadcasts to a [1,C,V] grid.
                valid_pair = chunk_code_valid[None, :, None] & valid_visits[
                    patient_idx
                ][None, None, :]
                chunk_masks = chunk_masks * valid_pair.to(chunk_masks.dtype)
                # valid_pair is [1,C,V], broadcast over all K phenotype masks.

                # Accumulate the per-visit KxK Gram matrix across all code rows.
                # This equals Gamma_j^T Gamma_j and supports L_distinct exactly.
                patient_gram = patient_gram + torch.einsum(
                    "kcv,lcv->vkl", chunk_masks, chunk_masks
                )

                # Save mask values only for augmented edges, forming Psi^k =
                # H_augmented * Gamma^k. All other scores remain available through
                # the Gram accumulation but are zero in the phenotype subgraphs.
                on_patient = edge_patients == patient_idx
                in_chunk = (edge_codes >= code_start) & (edge_codes < code_end)
                chosen_edges = on_patient & in_chunk
                if chosen_edges.any():
                    chosen_coordinates = edge_indices[chosen_edges]  # [E_chunk,3]: b, code, visit
                    local_codes = chosen_coordinates[:, 1] - code_start  # [E_chunk], rows in this chunk
                    local_visits = chosen_coordinates[:, 2]  # [E_chunk], visit columns
                    selected_masks = torch.stack(
                        [
                            chunk_masks[k, local_codes, local_visits]
                            for k in range(self.num_phenotypes)
                        ],
                        dim=-1,
                    )  # rows: selected links; columns: phenotype IDs 0..K-1
                    edge_coordinate_chunks.append(chosen_coordinates)
                    # Each row in selected_masks matches a row in chosen_coordinates:
                    # [patient, code, visit] -> K mask values.
                    edge_mask_chunks.append(selected_masks)

            all_patient_grams.append(patient_gram)

        # Stack each patient's sufficient statistics for the distinctness loss.
        mask_gram = torch.stack(all_patient_grams, dim=0)  # [B,V,K,K]

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
        subgraph_patients, subgraph_codes, subgraph_visits = subgraph_edge_indices.unbind(
            dim=1
        )
        num_subgraph_edges = subgraph_edge_indices.size(0)
        # phenotype_ids is [K,E]: each column is one link; rows list phenotype
        # IDs 0..K-1 so every link gets one sparse coordinate per phenotype.
        phenotype_ids = torch.arange(
            self.num_phenotypes, device=H_augmented.device
        ).unsqueeze(1).expand(-1, num_subgraph_edges)
        # Build sparse coordinate rows in matching K-major order:
        # [patient, phenotype, code, visit], one column per stored value.
        sparse_indices = torch.stack(
            [
                subgraph_patients.unsqueeze(0)
                .expand(self.num_phenotypes, -1)
                .reshape(-1),
                phenotype_ids.reshape(-1),
                subgraph_codes.unsqueeze(0)
                .expand(self.num_phenotypes, -1)
                .reshape(-1),
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
            raise ValueError("Expected phenotype graphs [B,K,N,V] and node_states [B,N,D]")

        batch_size, num_phenotypes, num_nodes, num_visits = phenotype_hypergraphs.shape
        if node_states.shape[:2] != (batch_size, num_nodes):
            raise ValueError("node_states batch/code dimensions must match phenotype graphs")
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
            edge_patients, edge_phenotypes, edge_codes, edge_visits = edge_coordinates.unbind(
                dim=0
            )
            flat_visit_ids = (
                (edge_patients * num_phenotypes + edge_phenotypes) * num_visits
                + edge_visits
            )
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
        flattened_visit_mask = valid_visits[:, None, :].expand(
            batch_size, num_phenotypes, num_visits
        ).reshape(batch_size * num_phenotypes, num_visits)

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
            flat_attention_weights = flat_attention_weights / flat_attention_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

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

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # STEP 1 — Decode the processed batch fields needed by this model.
        # The integer IDs in `codes` use exactly the input processor mapping used
        # above to create code_level_node_ids and therefore the rows of M.
        codes, _processor_mask = self._get_values_and_mask(kwargs[self.feature_key])
        codes = codes.to(self.device).long()
        batch_size = codes.size(0)

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

        # Temporary prediction scaffold: the phenotype vectors are now computed
        # and exposed below, but phenotype-level attention and diagnosis
        # prediction are the next stage. Until then, keep the earlier patient pool.
        observed_nodes = H_augmented.sum(dim=2) > 0  # [B, num_nodes]
        patient_repr = (node_states * observed_nodes.unsqueeze(-1)).sum(dim=1)
        patient_repr = patient_repr / observed_nodes.sum(dim=1).clamp_min(1).unsqueeze(-1)
        patient_repr = self.patient_activation(self.patient_projection(patient_repr))
        logits = self.classifier(patient_repr)

        result = {"logit": logits, "y_prob": self.prepare_y_prob(logits)}
        # Expose intermediate phenotype structures and temporal vectors on request.
        if kwargs.get("return_phenotypes", False):
            result["phenotype_hypergraphs"] = phenotype_hypergraphs
            result["phenotype_mask_gram"] = phenotype_mask_gram
            result["phenotype_edge_indices"] = phenotype_edge_indices
            result["phenotype_visit_embeddings"] = phenotype_visit_embeddings
            result["phenotype_vectors"] = phenotype_vectors
            result["temporal_attention_weights"] = temporal_attention_weights
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            result["y_true"] = y_true
            result["loss"] = self.get_loss_function()(logits, y_true)
        if kwargs.get("embed", False):
            result["embed"] = patient_repr
        return result
