"""Small synthetic check for SHy's patient-isolated hypergraph construction.

Run from this directory with either:
    python test_shy_hypergraph.py
    python -m pytest test_shy_hypergraph.py

These synthetic checks cover incidence construction, code -> visit -> code
message passing, and missing-diagnosis augmentation. They do not initialize a
full PyHealth dataset or test training, ICD-9 hierarchy lookups, phenotype
extraction, or prediction quality.
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


def make_augmentation_model(num_nodes: int = 7) -> SHy:
    """Create deterministic augmentation settings without loading a dataset."""
    model = make_hypergraph_builder(num_nodes)
    # Two original edges per patient and p=0.5 means exactly one new edge each.
    model.augmentation_ratio = 0.5
    model.num_similarity_heads = 1
    # A simple positive feature weighting makes hand-computed cosine scores clear.
    model.similarity_projections = nn.Parameter(torch.tensor([[1.0, 1.0]]))
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


def test_missing_diagnosis_augmentation_adds_only_valid_patient_local_edges() -> None:
    # Create two patients with two observed visits and one padded visit column.
    # The matrices are patient-specific: each patient has two original links.
    H = torch.zeros((2, 7, 3), dtype=torch.float32)
    H[0, 2, 0] = 1.0  # Patient 0: code 2 is in visit 0.
    H[0, 4, 1] = 1.0  # Patient 0: code 4 is in visit 1.
    H[1, 5, 0] = 1.0  # Patient 1: code 5 is in visit 0.
    H[1, 6, 1] = 1.0  # Patient 1: code 6 is in visit 1.
    # Visit column 2 is empty for both patients (batch padding).

    # Supply embeddings where we can predict the strongest missing links:
    # Patient 0's code 3 best matches visit 0; patient 1's code 2 best matches
    # visit 1. The patients intentionally have different visit embeddings.
    node_states = torch.zeros((2, 7, 2), dtype=torch.float32)
    node_states[0, 2] = torch.tensor([1.0, 0.0])
    node_states[0, 3] = torch.tensor([0.9, 0.1])
    node_states[0, 4] = torch.tensor([0.0, 1.0])
    node_states[1, 2] = torch.tensor([1.0, 0.0])
    node_states[1, 3] = torch.tensor([0.8, 0.2])
    node_states[1, 5] = torch.tensor([0.0, 1.0])
    node_states[1, 6] = torch.tensor([1.0, 0.0])

    visit_states = torch.zeros((2, 3, 2), dtype=torch.float32)
    visit_states[0, 0] = torch.tensor([1.0, 0.0])
    visit_states[0, 1] = torch.tensor([0.0, 1.0])
    visit_states[1, 0] = torch.tensor([0.0, 1.0])
    visit_states[1, 1] = torch.tensor([1.0, 0.0])
    # Visit column 2 stays zero and should never receive an augmented edge.

    model = make_augmentation_model()
    augmented_H, added_edges = model.augment_hypergraph(H, node_states, visit_states)

    # Each patient has two original edges, so floor(0.5 * 2) adds one link.
    assert added_edges.sum(dim=(1, 2)).tolist() == [1.0, 1.0]
    # The strongest pair is selected separately for each patient's embeddings.
    assert added_edges[0, 3, 0].item() == 1.0
    assert added_edges[1, 2, 1].item() == 1.0

    # No selected pair was already present, and the original H is preserved.
    assert torch.all((added_edges * H) == 0)
    assert torch.equal(augmented_H, H + added_edges)
    # Padding and unknown code rows (IDs 0 and 1) are never augmented.
    assert added_edges[:, 0:2, :].sum().item() == 0.0
    # Empty padded visit columns are not candidates for added connections.
    assert added_edges[:, :, 2].sum().item() == 0.0

    print("Missing-diagnosis augmentation checks passed.")
    print("New edges per patient:", added_edges.sum(dim=(1, 2)).tolist())


if __name__ == "__main__":
    test_patient_blocks_are_separate()
    test_message_passing_updates_codes_without_cross_patient_mixing()
    test_missing_diagnosis_augmentation_adds_only_valid_patient_local_edges()