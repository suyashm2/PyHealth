"""A PyHealth 2.1 compatible SHy starting point for next-visit diagnosis prediction.

The layer below implements visit-hypergraph message passing. It is an integration
scaffold, not a complete reproduction of every SHy objective (phenotype sampling,
false-negative completion, and explanation regularizers are not included).

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

For example, the histories could be ``[[[2, 3], [4, 0]], [[5, 0], [0, 0]]]``.
Here 0 is padding, 1 is the unknown-code token, and the numeric IDs for inputs
and labels can come from separate vocabularies. The example IDs are illustrative;
the fitted PyHealth processors assign the actual IDs for a dataset.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class SHy(BaseModel):
    """Predict a multi-label next-visit outcome from nested diagnosis histories.

    Expected input feature shape: ``[batch, visits, codes_per_visit]``.
    Token id 0 is padding. Labels are the processor's multi-hot target tensor,
    e.g. ``kwargs['next_conditions']`` with shape ``[batch, label_vocab_size]``.
    """

    def __init__(self,dataset: SampleDataset, embedding_dim: int = 64, hidden_dim: Optional[int] = None,) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError("SHy currently expects exactly one nested code feature")
        if len(self.label_keys) != 1:
            raise ValueError("SHy currently expects exactly one label key")
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]
        self.embedding_dim = embedding_dim
        hidden_dim = hidden_dim or embedding_dim

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.num_nodes = self.dataset.input_processors[self.feature_key].size()
        self.node_projection = nn.Linear(embedding_dim, hidden_dim)
        self.visit_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, self.get_output_size())

    def _get_values_and_mask(self, value) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unpack PyHealth's processor output, which may include a mask.

        A typical batch arrives as a ``value`` tensor shaped ``[B, V, C]``:
        patients, visits per patient, and code slots per visit. Some processors
        return that tensor by itself; others return a tuple containing ``value``
        and a padding ``mask``. This helper gives ``forward`` one consistent form.
        """
        if isinstance(value, torch.Tensor):
            return value, None
        schema = self.dataset.input_processors[self.feature_key].schema()
        tensor_values = value[schema.index("value")]
        mask = value[schema.index("mask")] if "mask" in schema else None
        return tensor_values, mask

    def construct_initial_hypergraph(
        self, codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return patient-isolated incidence ``H`` and valid-visit mask.

        ``H`` has shape ``[batch * num_nodes, batch * num_visits]``. Patient
        blocks are disjoint, so shared code IDs cannot pass messages across
        patients. Empty visits have no incident edges and are marked invalid.
        """
        if codes.dim() != 3:
            raise ValueError(
                f"Expected {self.feature_key} with shape [B,V,C], got {tuple(codes.shape)}"
            )
        codes = codes.long()
        batch_size, num_visits, _ = codes.shape
        device = codes.device
        # Example: a visit [2, 3, 0] has two recorded codes and one padding slot.
        # Zero is padding; IDs greater than zero (including unknown ID 1) count.
        valid_codes = (codes > 0) & (codes < self.num_nodes)
        valid_visits = valid_codes.any(dim=-1)

        # Flatten patient/code rows and patient/visit columns into separate blocks.
        # With B=2, patient 0 occupies the first num_nodes rows and patient 1 the
        # next num_nodes rows. Their columns are divided the same way. This is the
        # block-diagonal construction that prevents cross-patient messages.
        H = torch.zeros(
            batch_size * self.num_nodes,
            batch_size * num_visits,
            dtype=torch.float32,
            device=device,
        )
        b_idx, v_idx, c_idx = valid_codes.nonzero(as_tuple=True)
        if b_idx.numel():
            # Each valid (patient, visit, code) occurrence sets one incidence to 1.
            # A visit is a hyperedge; each code in it is one connected node.
            rows = b_idx * self.num_nodes + codes[b_idx, v_idx, c_idx]
            cols = b_idx * num_visits + v_idx
            H[rows, cols] = 1.0
        return H, valid_visits

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        codes, processor_mask = self._get_values_and_mask(kwargs[self.feature_key])
        codes = codes.to(self.device).long()

        # 2) Convert those histories into H, the patient-specific visit/code map.
        # valid_visits remembers which visit slots actually contain a code.
        H, valid_visits = self.construct_initial_hypergraph(codes)
        batch_size, num_visits, codes_per_visit = codes.shape

        # 3) PyHealth's embedding layer turns every input code ID into a D-vector.
        # The processor mask, when supplied, is forwarded in its expected format.
        masks = {}
        if processor_mask is not None:
            processor_mask = processor_mask.to(self.device)
            masks[self.feature_key] = processor_mask
        embedded = self.embedding_model(
            {self.feature_key: codes}, masks=masks
        )[self.feature_key]
        if embedded.dim() != 4:
            raise ValueError(
                "SHy expects EmbeddingModel output [B,V,C,D] for a nested code feature; "
                f"got {tuple(embedded.shape)}"
            )

        # 4) Pool code occurrences into node features, separately for each patient.
        # If a code appears in multiple visits, average its embeddings for that
        # patient's node. Padding and invalid IDs contribute nothing. The same
        # diagnosis in two patients gets two separate rows due to patient_node_ids.
        valid = (codes > 0) & (codes < self.num_nodes)
        flat_patient = torch.arange(batch_size, device=self.device)[:, None, None]
        patient_node_ids = flat_patient * self.num_nodes + codes.clamp(0, self.num_nodes - 1)
        x = embedded.new_zeros((batch_size * self.num_nodes, self.embedding_dim))
        counts = embedded.new_zeros((batch_size * self.num_nodes, 1))
        if valid.any():
            flat_ids = patient_node_ids[valid]
            x.index_add_(0, flat_ids, embedded[valid])
            counts.index_add_(0, flat_ids, embedded.new_ones((flat_ids.numel(), 1)))
        x = x / counts.clamp_min(1.0)
        x = self.activation(self.node_projection(x))

        # 5) Send information code -> visit -> code using H. Dividing by degrees
        # averages the connected nodes/visits instead of letting large visits or
        # frequently repeated nodes dominate just because they have more links.
        edge_degree = H.sum(dim=0).clamp_min(1.0)
        node_degree = H.sum(dim=1).clamp_min(1.0)
        visit_repr = (H.transpose(0, 1) @ x) / edge_degree.unsqueeze(1)
        visit_repr = self.activation(self.visit_projection(visit_repr))
        node_repr = (H @ visit_repr) / node_degree.unsqueeze(1)
        visit_repr = (H.transpose(0, 1) @ node_repr) / edge_degree.unsqueeze(1)
        visit_repr = visit_repr.view(batch_size, num_visits, -1)

        # 6) Fold each patient's visit vectors into one patient vector. Empty padded
        # visits are masked out. The clamp makes a completely empty history produce
        # zeros rather than a divide-by-zero/NaN.
        visit_weights = valid_visits.to(visit_repr.dtype).unsqueeze(-1)
        patient_repr = (visit_repr * visit_weights).sum(dim=1)
        patient_repr = patient_repr / visit_weights.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(patient_repr)

        # 7) Each logit position corresponds to the label processor's vocabulary.
        # Thus input IDs and label IDs may be numbered differently; the learned
        # classifier connects the history representation to the target positions.
        result = {"logit": logits, "y_prob": self.prepare_y_prob(logits)}
        if self.label_key in kwargs:
            # Training/evaluation batches include the multi-hot target, such as
            # next_conditions. PyHealth's configured loss compares it to logits.
            y_true = kwargs[self.label_key].to(self.device)
            result["y_true"] = y_true
            result["loss"] = self.get_loss_function()(logits, y_true)
        if kwargs.get("embed", False):
            result["embed"] = patient_repr
        return result
