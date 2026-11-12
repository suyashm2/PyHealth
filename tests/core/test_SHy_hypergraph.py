"""Small synthetic check for SHy's patient-isolated hypergraph construction.

Run from this directory with either:
    python test_shy_hypergraph.py
    python -m pytest test_shy_hypergraph.py

These synthetic checks cover incidence construction, code -> visit -> code
message passing, missing-diagnosis augmentation, and phenotype extraction. They
do not initialize a full PyHealth dataset, test ICD-9 hierarchy lookups,
or assess prediction quality.
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


def make_phenotype_extraction_model(num_nodes: int = 7, num_phenotypes: int = 3) -> SHy:
    """Create small independent mask MLPs for a synthetic extraction check."""
    model = make_hypergraph_builder(num_nodes)
    model.num_phenotypes = num_phenotypes
    model.phenotype_temperature = 1.0
    model.phenotype_code_chunk_size = 2
    # The scorer input is [code embedding || visit embedding], each of width 2.
    model.phenotype_extractors = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 1))
            for _ in range(num_phenotypes)
        ]
    )
    return model


def make_temporal_embedding_model(
    embedding_dim: int = 2, hidden_dim: int = 3
) -> SHy:
    """Create the GRU and temporal-attention modules without a real dataset."""
    model = make_hypergraph_builder(num_nodes=4)
    model.num_phenotypes = 2
    model.hidden_dim = hidden_dim
    model.phenotype_gru = nn.GRU(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        batch_first=True,
    )
    model.temporal_attention = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )
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


def test_phenotype_extraction_masks_only_augmented_patient_edges() -> None:
    # Build a tiny augmented graph: two patients, seven code rows, three visits.
    # Visit 2 is padding for both patients; code rows 0 and 1 are PAD and UNK.
    H_augmented = torch.zeros((2, 7, 3), dtype=torch.float32)
    H_augmented[0, 2, 0] = 1.0
    H_augmented[0, 3, 0] = 1.0  # Added/selected diagnosis for patient 0.
    H_augmented[0, 4, 1] = 1.0
    H_augmented[1, 5, 0] = 1.0
    H_augmented[1, 6, 1] = 1.0
    H_augmented[1, 2, 1] = 1.0  # Added/selected diagnosis for patient 1.

    # Give each diagnosis and visit a small, two-feature embedding. The MLPs score
    # every valid code/visit pair, but only H~ links can survive as subgraph edges.
    node_states = torch.randn((2, 7, 2), dtype=torch.float32)
    visit_states = torch.randn((2, 3, 2), dtype=torch.float32)
    model = make_phenotype_extraction_model(num_phenotypes=3)

    # Zero weights and biases make every logit exactly zero. At test time the
    # deterministic 0.5 threshold therefore retains every valid code/visit pair.
    # This makes the mask Gram matrix reveal that even code rows absent from H~
    # were scored, while the phenotype subgraphs still contain only H~ edges.
    for extractor in model.phenotype_extractors:
        for parameter in extractor.parameters():
            nn.init.zeros_(parameter)

    # Evaluation is deterministic: logits are thresholded at probability 0.5.
    model.eval()
    phenotype_graphs, mask_gram, edge_indices = model.extract_phenotypes(
        H_augmented, node_states, visit_states
    )
    assert phenotype_graphs.layout == torch.sparse_coo
    assert tuple(phenotype_graphs.shape) == (2, 3, 7, 3)
    assert edge_indices.shape == (6, 3), edge_indices.shape
    assert mask_gram.shape == (2, 3, 3, 3), mask_gram.shape

    # Densifying is only for these small assertions. In the model, masks remain
    # sparse so storing K phenotype graphs does not allocate K copies of all zeros.
    dense_phenotypes = phenotype_graphs.to_dense()
    assert set(dense_phenotypes.unique().tolist()).issubset({0.0, 1.0})

    # Every selected phenotype edge must already exist in that patient's H~.
    # This also ensures the mask cannot connect one patient's code to another's visit.
    allowed_edges = H_augmented.unsqueeze(1).expand_as(dense_phenotypes) > 0
    assert torch.all((dense_phenotypes == 0) | allowed_edges)
    # Since zero logits produce all-one masks, each valid visit's KxK Gram
    # entries equal the number of valid diagnosis rows (IDs 2 through 6 = five).
    # This proves the extractor also scored code rows absent from that patient's H~.
    assert torch.all(mask_gram[:, 0] == 5.0)
    assert torch.all(mask_gram[:, 1] == 5.0)
    assert torch.all(mask_gram[:, 2] == 0.0)
    # Those full-grid masks are filtered by H~: with all-one masks, each phenotype
    # subgraph must exactly match its patient's augmented incidence matrix.
    assert torch.equal(dense_phenotypes, H_augmented.unsqueeze(1).expand_as(dense_phenotypes))

    # PAD/UNK rows and padded visit columns were absent from H~, so they must
    # remain absent from every one of the K phenotype-specific sub-hypergraphs.
    assert dense_phenotypes[:, :, 0:2, :].sum().item() == 0.0
    assert dense_phenotypes[:, :, :, 2].sum().item() == 0.0

    # During training, hard Gumbel masks have binary forward values but retain a
    # straight-through gradient, allowing the phenotype MLPs to learn their masks.
    model.train()
    # Reinitialize after the deterministic all-one checks so gradients flow
    # through nontrivial MLP outputs in this separate training-mode check.
    for parameter in model.phenotype_extractors.parameters():
        nn.init.normal_(parameter, mean=0.0, std=0.1)
    sampled_graphs, sampled_gram, _ = model.extract_phenotypes(
        H_augmented, node_states, visit_states
    )
    (sampled_graphs.to_dense().sum() + sampled_gram.sum()).backward()
    final_layer = model.phenotype_extractors[0][-1]
    assert final_layer.weight.grad is not None
    assert torch.isfinite(final_layer.weight.grad).all()

    print("Phenotype extraction checks passed.")
    print("Sparse phenotype graph shape:", tuple(phenotype_graphs.shape))
    print("Scored augmented edges:", edge_indices.shape[0])


def test_phenotype_visit_embeddings_and_temporal_attention() -> None:
    # Sparse Ψ for two patients and two phenotypes. Coordinates are
    # [patient, phenotype, code, visit].
    sparse_coordinates = torch.tensor(
        [
            [0, 0, 0, 0, 1],  # patient IDs
            [0, 0, 0, 1, 0],  # phenotype IDs
            [2, 3, 2, 3, 2],  # diagnosis-code IDs
            [0, 0, 1, 1, 0],  # visit IDs
        ],
        dtype=torch.long,
    )
    phenotype_graphs = torch.sparse_coo_tensor(
        sparse_coordinates,
        torch.ones(5),
        size=(2, 2, 4, 3),
    ).coalesce()

    # Patient 0: code 2=[1,2], code 3=[3,4]. Patient 1 has its own code-2
    # vector [10,20]. These make Equation 9's per-visit sums easy to verify.
    node_states = torch.zeros((2, 4, 2), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        node_states[0, 2] = torch.tensor([1.0, 2.0])
        node_states[0, 3] = torch.tensor([3.0, 4.0])
        node_states[1, 2] = torch.tensor([10.0, 20.0])
    valid_visits = torch.tensor(
        [[True, True, False], [True, False, False]], dtype=torch.bool
    )

    model = make_temporal_embedding_model()
    phenotype_vectors, attention_weights, visit_embeddings = (
        model.embed_phenotypes_over_time(
            phenotype_graphs,
            node_states,
            valid_visits=valid_visits,
        )
    )

    # Shapes: vectors [B,K,H], attention [B,K,V], and Equation 9 sums [B,K,V,D].
    assert phenotype_vectors.shape == (2, 2, 3), phenotype_vectors.shape
    assert attention_weights.shape == (2, 2, 3), attention_weights.shape
    assert visit_embeddings.shape == (2, 2, 3, 2), visit_embeddings.shape

    # Equation 9 sums linked diagnosis vectors: for patient 0, phenotype 0's
    # visit 0 is code2 + code3 = [4,6], and visit 1 is code2 = [1,2].
    assert torch.equal(visit_embeddings[0, 0, 0], torch.tensor([4.0, 6.0]))
    assert torch.equal(visit_embeddings[0, 0, 1], torch.tensor([1.0, 2.0]))
    # Phenotype 1 contains code 3 in visit 1 only.
    assert torch.equal(visit_embeddings[0, 1, 1], torch.tensor([3.0, 4.0]))
    # Patient 1's matching code ID uses its own embedding, with no cross-patient mix.
    assert torch.equal(visit_embeddings[1, 0, 0], torch.tensor([10.0, 20.0]))
    assert torch.equal(visit_embeddings[1, 1], torch.zeros((3, 2)))

    # Attention normalizes over valid visits and assigns no weight to padding.
    assert torch.allclose(attention_weights[0, :, :2].sum(dim=-1), torch.ones(2))
    assert torch.allclose(attention_weights[1, 0, 0], torch.tensor(1.0))
    assert torch.all(attention_weights[0, :, 2] == 0)
    assert torch.all(attention_weights[1, :, 1:] == 0)
    assert torch.isfinite(phenotype_vectors).all()

    # Fully empty histories have zero attention and zero phenotype vectors,
    # rather than NaNs from softmax over a row containing only padded visits.
    no_valid_visits = torch.zeros_like(valid_visits)
    empty_phenotype_graphs = torch.sparse_coo_tensor(
        torch.empty((4, 0), dtype=torch.long),
        torch.empty((0,), dtype=torch.float32),
        size=(2, 2, 4, 3),
    ).coalesce()
    empty_vectors, empty_attention, _ = model.embed_phenotypes_over_time(
        empty_phenotype_graphs,
        node_states,
        valid_visits=no_valid_visits,
    )
    assert torch.equal(empty_vectors, torch.zeros_like(empty_vectors))
    assert torch.equal(empty_attention, torch.zeros_like(empty_attention))

    # The GRU/attention representation remains differentiable with respect to M.
    phenotype_vectors.sum().backward()
    assert node_states.grad is not None
    assert torch.isfinite(node_states.grad).all()

    print("Temporal phenotype embedding checks passed.")
    print(
        "Phenotype vector shape [patients, phenotypes, hidden]:",
        tuple(phenotype_vectors.shape),
    )
    print(
        "Temporal attention shape [patients, phenotypes, visits]:",
        tuple(attention_weights.shape),
    )


if __name__ == "__main__":
    test_patient_blocks_are_separate()
    test_message_passing_updates_codes_without_cross_patient_mixing()
    test_missing_diagnosis_augmentation_adds_only_valid_patient_local_edges()
    test_phenotype_extraction_masks_only_augmented_patient_edges()
    test_phenotype_visit_embeddings_and_temporal_attention()