"""Small synthetic check for SHy's patient-isolated hypergraph construction.

Run from this directory with either:
    python test_shy_hypergraph.py
    python -m pytest test_shy_hypergraph.py

These synthetic checks cover incidence construction and the code -> visit -> code
message-passing step. They do not initialize a full PyHealth dataset or test
training, ICD-9 hierarchy lookups, phenotype extraction, or prediction quality.
"""
import torch
import torch.nn as nn

from pyhealth.models.SHy import SHy


def make_hypergraph_builder(num_nodes: int = 6) -> SHy:
    """Create just enough SHy state to call construct_initial_hypergraph."""
    # Bypass SHy.__init__: graph construction only needs these two attributes.
    model = object.__new__(SHy)
    nn.Module.__init__(model)
    model.feature_key = "conditions"
    model.num_nodes = num_nodes
    model.pad_idx = 0
    model.unk_idx = 1
    return model


def make_message_passing_model(num_nodes: int = 6) -> SHy:
    """Create a one-layer identity UniGIN update for transparent expected values."""
    model = make_hypergraph_builder(num_nodes)
    # Identity weights and zero epsilon make the update equal to:
    # old code vector + the mean vector of each visit containing that code.
    model.message_layers = nn.ModuleList([nn.Identity()])
    model.message_eps = nn.ParameterList([nn.Parameter(torch.tensor(0.0))])
    model.message_activation = nn.LeakyReLU()
    return model


def test_patient_blocks_are_separate() -> None:
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

    # H has one [code rows, visit columns] matrix per patient.
    assert incidence.shape == (2, 6, 2), incidence.shape
    assert valid_visits.tolist() == [[True, True], [True, False]]

    # Patient 0: visit 0 connects codes 2,3; visit 1 connects code 4.
    assert incidence[0, 2, 0].item() == 1.0
    assert incidence[0, 3, 0].item() == 1.0
    assert incidence[0, 4, 1].item() == 1.0

    # Patient 1 has its own matrix: visit 0 connects code 5, visit 1 is padding.
    assert incidence[1, 5, 0].item() == 1.0
    assert incidence[1, :, 1].sum().item() == 0.0

    # The shared diagnosis ID 5 is represented independently for each patient.
    # Patient 0 did not have code 5, while patient 1 does.
    assert incidence[0, 5, :].sum().item() == 0.0
    assert incidence[1, 5, 0].item() == 1.0

    # Unknown ID 1 is not a diagnosis node. An unknown-only visit has no links
    # and is therefore marked invalid, just like a padding-only visit.
    unknown_only = torch.tensor([[[1, 0]]], dtype=torch.long)
    unknown_H, unknown_valid_visits = model.construct_initial_hypergraph(unknown_only)
    assert unknown_H.shape == (1, 6, 1), unknown_H.shape
    assert unknown_H.sum().item() == 0.0
    assert unknown_valid_visits.tolist() == [[False]]

    print("Hypergraph construction checks passed.")
    print("Batched incidence shape [patients, codes, visits]:", tuple(incidence.shape))
    print("Valid visits:", valid_visits.tolist())
    for patient_idx, patient_H in enumerate(incidence):
        print(f"Patient {patient_idx} incidence matrix:\n", patient_H)


def test_message_passing_updates_codes_without_cross_patient_mixing() -> None:
    # Patient 0 has visit {2,3} and visit {4}; patient 1 has visit {5}.
    codes = torch.tensor(
        [
            [[2, 3], [4, 0]],
            [[5, 0], [0, 0]],
        ],
        dtype=torch.long,
    )
    model = make_message_passing_model(num_nodes=6)
    H, _ = model.construct_initial_hypergraph(codes)

    # One-dimensional initial M makes the arithmetic easy to follow.
    # Patient 0's first visit averages (2 + 4) / 2 = 3; its second is 8.
    # Patient 1's visit has only code 5, so its visit vector is 10.
    initial_M = torch.tensor([[0.0], [0.0], [2.0], [4.0], [8.0], [10.0]])
    updated = model.message_passing(H, initial_M)

    assert updated.shape == (2, 6, 1), updated.shape
    # Code 2 receives visit-0's mean: 2 + 3 = 5.
    assert torch.isclose(updated[0, 2, 0], torch.tensor(5.0))
    # Code 3 receives the same visit mean: 4 + 3 = 7.
    assert torch.isclose(updated[0, 3, 0], torch.tensor(7.0))
    # Code 4 receives visit-1's mean: 8 + 8 = 16.
    assert torch.isclose(updated[0, 4, 0], torch.tensor(16.0))
    # Code 5 in patient 1 receives only patient 1's visit: 10 + 10 = 20.
    assert torch.isclose(updated[1, 5, 0], torch.tensor(20.0))
    # Patient 1 did not have code 2, so patient 0's message cannot update it.
    assert torch.isclose(updated[1, 2, 0], torch.tensor(2.0))
    print('Message Passing passed!')


if __name__ == "__main__":
    test_patient_blocks_are_separate()
    test_message_passing_updates_codes_without_cross_patient_mixing()
