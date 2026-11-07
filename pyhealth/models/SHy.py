"""A PyHealth 2.1 compatible SHy starting point for next-visit diagnosis prediction.

The model builds hierarchy-aware initial code embeddings and applies a batched
UniGIN-style code -> visit -> code message-passing step on patient-isolated
hypergraphs. It is not yet a complete SHy reproduction: false-negative edge
completion, temporal phenotype extraction, and the paper's explanation objectives
are not implemented.

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
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError("SHy currently expects exactly one nested code feature")
        if len(self.label_keys) != 1:
            raise ValueError("SHy currently expects exactly one label key")
        if num_hierarchy_levels <= 0:
            raise ValueError("num_hierarchy_levels must be positive")
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

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]
        self.embedding_dim = embedding_dim
        self.num_hierarchy_levels = num_hierarchy_levels
        self.per_level_dim = embedding_dim // num_hierarchy_levels
        self.augmentation_ratio = float(augmentation_ratio)
        self.num_similarity_heads = num_similarity_heads
        hidden_dim = hidden_dim or embedding_dim

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
        # STEP 2 — Resolve every input diagnosis to a fixed-width ICD-9 path.
        # 2a. Standardize strings such as "4280" to the hierarchy's "428.0".
        # 2b. Ask PyHealth's ICD9CM map for the code's ancestors.
        # 2c. Drop ICD-9's universal root, keep the most-specific levels when a
        #     path exceeds L, and add code-specific virtual children when shorter.
        # 2d. Assign a level-local integer ID to every hierarchy node. Special
        #     tokens get their own IDs and will be masked from graph operations.
        # ---------------------------------------------------------------------
        icd9cm = InnerMap.load("ICD9CM")
        code_level_node_ids, level_vocab_sizes, unmapped = self._make_level_node_ids(
            self.id_to_code, icd9cm, num_hierarchy_levels
        )
        if unmapped:
            warnings.warn(
                f"{len(unmapped)} input tokens were not found in the ICD-9-CM "
                "hierarchy. They receive code-specific virtual hierarchy nodes.",
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

        # The paper's temporal phenotype predictor is not implemented yet. This
        # pooling/classifier head is a temporary downstream prediction scaffold.
        self.patient_projection = nn.Linear(embedding_dim, hidden_dim)
        self.patient_activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, self.get_output_size())

    @staticmethod
    def _make_level_node_ids(
        id_to_code: Sequence[Optional[str]], icd9cm: Any, num_levels: int
    ) -> Tuple[torch.Tensor, List[int], List[str]]:
        """Map each processor ID to one hierarchy-node ID at every level."""
        level_vocabularies: List[Dict[str, int]] = [dict() for _ in range(num_levels)]
        rows: List[List[int]] = []
        unmapped: List[str] = []
        special_tokens = {"<pad>", "<unk>"}

        for raw_token in id_to_code:
            token = str(raw_token)
            if token in special_tokens:
                # Special rows remain well-defined but are excluded by the code mask.
                path = [f"__special__:{token}:level:{level}" for level in range(num_levels)]
            else:
                standardized = icd9cm.standardize(token)
                try:
                    # PyHealth returns ancestors nearest-first; reverse to root-first.
                    ancestors = list(icd9cm.get_ancestors(standardized))
                    path = list(reversed(ancestors)) + [standardized]
                    # ICD9CM includes its universal 001-999.99 root in paths.
                    # SHy's four disease levels start below that dataset-wide root.
                    if path and path[0] == "001-999.99":
                        path = path[1:]
                except (KeyError, ValueError):
                    # Preserve an embedding for a token absent from the ontology.
                    path = [standardized]
                    unmapped.append(token)

                if len(path) > num_levels:
                    # Keep the most-specific levels, including the diagnosis itself.
                    path = path[-num_levels:]
                elif len(path) < num_levels:
                    # A code may be a non-leaf or have no known ancestors. Add
                    # code-specific virtual descendants so every row has L levels.
                    path.extend(
                        f"__virtual_child__:{standardized}:{depth}"
                        for depth in range(num_levels - len(path))
                    )

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

        # Temporary downstream pooling: average updated diagnosis vectors that
        # occur in the augmented history; empty histories become zeros. Later
        # phenotype extraction should consume H_augmented directly.
        observed_nodes = H_augmented.sum(dim=2) > 0  # [B, num_nodes]
        patient_repr = (node_states * observed_nodes.unsqueeze(-1)).sum(dim=1)
        patient_repr = patient_repr / observed_nodes.sum(dim=1).clamp_min(1).unsqueeze(-1)
        patient_repr = self.patient_activation(self.patient_projection(patient_repr))
        logits = self.classifier(patient_repr)

        result = {"logit": logits, "y_prob": self.prepare_y_prob(logits)}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            result["y_true"] = y_true
            result["loss"] = self.get_loss_function()(logits, y_true)
        if kwargs.get("embed", False):
            result["embed"] = patient_repr
        return result
