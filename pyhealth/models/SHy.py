from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from pyhealth.datasets import Dataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_embedding_module

class SHy(BaseModel):
    """Self-Explaining Hypergraph Neural Network (SHy) for EHR prediction.

    This model constructs hypergraphs from clinical encounters and applies 
    higher-order hypergraph convolutions for clinical prediction. The goal 
    is given hypergraph containing all the previous visits and diagnoses to 
    predict the diagnoses at the next visit. """

    '''Example after processing the dataset class (contains all the relevant information we need)
         Example:
                {
                    "patient_id": "10006",
                    "visit_id": ["v_001", "v_002", "v_003"],
                    "conditions": [
                        ["4019", "25000"],  # Visit 1 codes
                        ["25000", "4280"],   # Visit 2 codes
                        ["4019"]             # Visit 3 codes
                    ],
                    "label": ["4280", "2765"] # The target codes to predict for the next visit
                } 
    '''

    def __init__(self, dataset: Dataset, feature_keys: List[str], label_key: str, mode: str, emb_size: int = 64, top_k_latent: int = 3) -> None:
        super().__init__(
            dataset=dataset, # Dataset object len(p) where p is number of patients and each element is a dict like example
            feature_keys=feature_keys, # list of what columns in dataset to look at (i.e. co)
            label_key=label_key, # what we want to predict
            mode=mode, # multilabel for this model (next visit can have multiple conditions)
        )
        self.emb_size = emb_size # how many numbers to represent each medical code

        # Standard embedding module (or your partner's hierarchical embedding module)
        self.embedding_module = get_embedding_module(
            feature_keys=self.feature_keys,
            dataset=self.dataset,
            emb_size=self.emb_size,
        )

        # Get the number of total diagnosis codes -- that will set up the number of rows in the incidence matrix
        self.num_nodes = self.dataset.get_vocab_size(self.feature_keys[0]) 

        # Final classification head and loss function
        self.fc = nn.Linear(self.emb_size, self.get_output_size())
        self.loss_function = self.get_loss_function()

    def construct_initial_hypergraph(self, patient_batch: Dict[str, torch.Tensor],device: torch.device,) -> Tuple[torch.Tensor, int]:
        """Constructs the initial hypergraph incidence matrix H for a batch of patients."""
        feature_key = self.feature_keys[0] # which column from dataset we want
        sequences = patient_batch[feature_key]  # [batch_size, num_visits, num_diagnoses]

        row_indices = []
        col_indices = []
        total_hyperedges = 0

        for patient_visits in sequences: # extracts one patient's set of visits and medical codes
            '''
            Example: 
            [
                [code 1, code 2] # visit 1
                ,[code 3] # visit 4
            ]
            '''
            for visit_codes in patient_visits: # extracts one visit and the set of codes 
                '''
                Example:
                [code 1, code 2] # visit 1
                '''
                hyperedge_idx = total_hyperedges # counts the visit number for column initialization
                total_hyperedges += 1
                
                for code_idx in visit_codes:
                    if code_idx < self.num_nodes:
                        row_indices.append(code_idx) # represents each diagnosis code
                        col_indices.append(hyperedge_idx) # represents the visit number

        # edge case where there are 0 visits return one visit all with 0s
        if total_hyperedges == 0:
            return torch.zeros((self.num_nodes, 1), device=device), 1

        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device) # row 0 = diagnosis code, row 1 = visit indicies (for sparse index processing)
        values = torch.ones(len(row_indices), dtype=torch.float32, device=device)

        # has all diagnosis codes and different patient visits concatanated side by side 
        H = torch.sparse_coo_tensor(
            indices, values, size=(self.num_nodes, total_hyperedges)
        ).to_dense()

        return H, total_hyperedges


    def forward(self,patient_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for the SHy model."""
        device = self.fc.weight.device

        # 1. Obtain node embeddings (X) : for encoding the diagnosis -- the row is the diagnosis, column is embeddings size
        embeddings = self.embedding_module(patient_batch)   # embeddings for each diagnosis
        if isinstance(embeddings, dict):
            X = embeddings[self.feature_keys[0]]
        else:
            X = embeddings

        # If X is batched [batch_size, seq_len, emb_size], we can pool or extract unique code embeddings.
        # Let's flatten/collapse appropriately for graph convolution:
        if X.dim() > 2:
            X = X.reshape(-1, self.emb_size)
        
        # Ensure X matches vocabulary size if required by your convolution layers
        if X.shape[0] != self.num_nodes:
            # Padding or slicing fallback if shape differs
            if X.shape[0] < self.num_nodes:
                padding = torch.zeros((self.num_nodes - X.shape[0], self.emb_size), device=device)
                X = torch.cat([X, padding], dim=0)
            else:
                X = X[:self.num_nodes, :]

        # 2. Construct the initial hypergraph H for this batch and handle incompleteness (unrecorded conditions)
        H, visit_len = self.construct_initial_hypergraph(patient_batch, device=device)

        # 3. Next step: Pass X and H into your message passing (HGNN) layer!
        # (We will wire up the message passing and phenotype extraction next)
        patient_repr = torch.mean(X, dim=0, keepdim=True).expand(len(patient_batch['patient_id']), -1)
        logits = self.fc(patient_repr)

        # PyHealth mandatory return structure
        y_true = self.prepare_y_true(patient_batch)
        y_prob = self.prepare_y_prob(logits)
        loss = self.loss_function(logits, y_true)

