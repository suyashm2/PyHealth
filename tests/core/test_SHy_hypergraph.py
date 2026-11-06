"""Small synthetic check for SHy's patient-isolated hypergraph construction.

Run from this directory with:
    python test_shy_hypergraph.py

This checks only the history-to-incidence-matrix step. It does not initialize a
full PyHealth dataset or test training, embeddings, or prediction quality.
"""
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyhealth.models.SHy import SHy

def make_hypergraph_builder(num_nodes: int = 6) -> SHy:
    """Create just enough SHy state to call construct_initial_hypergraph."""
    # Bypass SHy.__init__: graph construction only needs these two attributes.
    model = object.__new__(SHy)
    nn.Module.__init__(model)
    model.feature_key = "conditions"
    model.num_nodes = num_nodes
    return model


def main() -> None:
    # Two patients, each with two padded visit slots and two padded code slots.
    # Patient 0: visit 0 has codes 2 and 3; visit 1 has code 4.
    # Patient 1: visit 0 has code 5; visit 1 is entirely padding.
    codes = torch.tensor(
        [
            [[2, 3], [4, 0]],
            [[5, 0], [0, 0]],
        ],
        dtype=torch.long,
    )

    model = make_hypergraph_builder(num_nodes=6)
    incidence, valid_visits = model.construct_initial_hypergraph(codes)

    # There are B*num_nodes rows and B*num_visits columns.
    assert incidence.shape == (12, 4), incidence.shape
    assert valid_visits.tolist() == [[True, True], [True, False]]

    # Patient 0's visit columns: column 0 connects codes 2,3; column 1 connects 4.
    assert incidence[2, 0].item() == 1.0
    assert incidence[3, 0].item() == 1.0
    assert incidence[4, 1].item() == 1.0

    # Patient 1 begins at row offset 6 and column offset 2.
    assert incidence[6 + 5, 2].item() == 1.0
    assert incidence[:, 3].sum().item() == 0.0  # all-padding visit

    # The two patient blocks must have no cross-links.
    assert incidence[:6, 2:].sum().item() == 0.0
    assert incidence[6:, :2].sum().item() == 0.0

    print("Hypergraph construction checks passed.")
    print("Incidence shape:", tuple(incidence.shape))
    print("Valid visits:", valid_visits.tolist())
    print("Incidence matrix:\n", incidence)


if __name__ == "__main__":
    main()
