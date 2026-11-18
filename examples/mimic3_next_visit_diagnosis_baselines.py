"""Next-visit diagnosis prediction on MIMIC-III with PyHealth baseline models.

Trains RNN, RETAIN and Transformer models on the NextVisitDiagnosisMIMIC3 task
and reports Recall@k and nDCG@k, the metrics used by the SHy paper (Yu et al.,
CHIL 2025). These baselines are the comparison point for the SHy model.

The pipeline follows PyHealth's five steps:
    1. Load the dataset.
    2. Set the task (one sample per patient: earlier visits predict the last).
    3. Split by patient into train, validation and test sets.
    4. Train each model with the PyHealth Trainer.
    5. Evaluate on the test set with Recall@k and nDCG@k.

By default the script uses PyHealth's public synthetic MIMIC-III data, so it
runs without credentials. To use the real MIMIC-III 1.4 data, pass the folder
that holds its CSV files:

    python examples/mimic3_next_visit_diagnosis_baselines.py \\
        --root /path/to/mimic-iii/1.4

Use --sample-per-visit to make one sample for every visit after the first
instead of one sample per patient.

Example output on the synthetic MIMIC-III data (2,085 samples, 20 epochs, seed
42, one NVIDIA RTX 4070 laptop GPU, about 2 minutes):

    model         Recall@10    nDCG@10  Recall@20    nDCG@20
    rnn              0.2036     0.2047     0.2847     0.2268
    retain           0.1933     0.1874     0.2719     0.2091
    transformer      0.1974     0.1984     0.2728     0.2182

Random guessing would give a Recall@20 of about 0.01 (20 of about 1,700
codes). Scores on synthetic data are lower than those reported on real
MIMIC-III and are meant only to show that the pipeline works.
"""

import argparse
import math

import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RETAIN, RNN, Transformer
from pyhealth.tasks import NextVisitDiagnosisMIMIC3
from pyhealth.trainer import Trainer

SYNTHETIC_MIMIC3_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/"
MODEL_CLASSES = {"rnn": RNN, "retain": RETAIN, "transformer": Transformer}
TOP_K_VALUES = [10, 20]


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Computes Recall@k averaged over samples, as in the SHy authors' code.

    For each sample, Recall@k is the number of true codes among the k
    highest-scored codes, divided by the number of true codes.

    Args:
        y_true: 0/1 matrix of true codes, shape (num_samples, num_codes).
        y_prob: Predicted scores, same shape as ``y_true``.
        k: Number of top-scored codes to consider.

    Returns:
        The mean Recall@k over samples that have at least one true code.
    """
    recalls = []
    for true_row, prob_row in zip(y_true, y_prob):
        true_codes = set(np.flatnonzero(true_row))
        if len(true_codes) == 0:
            continue
        top_k_codes = set(np.argsort(-prob_row)[:k])
        recalls.append(len(true_codes & top_k_codes) / len(true_codes))
    return float(np.mean(recalls))


def ndcg_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Computes nDCG@k averaged over samples, as in the SHy authors' code.

    Each true code found at rank position i (counting from 0) among the k
    highest-scored codes adds 1 / log2(i + 2). The sum is divided by the best
    possible sum, so a perfect ranking scores 1.

    Args:
        y_true: 0/1 matrix of true codes, shape (num_samples, num_codes).
        y_prob: Predicted scores, same shape as ``y_true``.
        k: Number of top-scored codes to consider.

    Returns:
        The mean nDCG@k over samples that have at least one true code.
    """
    scores = []
    for true_row, prob_row in zip(y_true, y_prob):
        num_true_codes = int(true_row.sum())
        if num_true_codes == 0:
            continue
        top_k_codes = np.argsort(-prob_row)[:k]
        dcg = sum(
            1 / math.log2(position + 2)
            for position, code in enumerate(top_k_codes)
            if true_row[code] == 1
        )
        ideal_dcg = sum(
            1 / math.log2(position + 2) for position in range(min(k, num_true_codes))
        )
        scores.append(dcg / ideal_dcg)
    return float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    """Reads the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=SYNTHETIC_MIMIC3_ROOT)
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_CLASSES), choices=MODEL_CLASSES
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sample-per-visit", action="store_true")
    parser.add_argument("--dev", action="store_true", help="use a small subset")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Runs the five-step pipeline for each baseline model."""
    args = parse_args()

    # STEP 1: load the diagnosis table
    base_dataset = MIMIC3Dataset(root=args.root, tables=["DIAGNOSES_ICD"], dev=args.dev)
    base_dataset.stats()

    # STEP 2: set the next-visit diagnosis task
    task = NextVisitDiagnosisMIMIC3(sample_per_visit=args.sample_per_visit)
    sample_dataset = base_dataset.set_task(task)
    print(f"Samples: {len(sample_dataset)}")

    # STEP 3: split by patient, so no patient is in more than one set
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.7, 0.1, 0.2], seed=args.seed
    )
    train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, args.batch_size, shuffle=False)

    results = {}
    for model_name in args.models:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # STEP 4: train, keeping the epoch with the best validation PR-AUC
        model = MODEL_CLASSES[model_name](dataset=sample_dataset)
        trainer = Trainer(model=model, metrics=["pr_auc_samples"])
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params={"lr": args.learning_rate},
            monitor="pr_auc_samples",
        )

        # STEP 5: evaluate on the test set with the paper's metrics
        y_true, y_prob, _ = trainer.inference(test_loader)
        results[model_name] = {}
        for k in TOP_K_VALUES:
            results[model_name][f"Recall@{k}"] = recall_at_k(y_true, y_prob, k)
            results[model_name][f"nDCG@{k}"] = ndcg_at_k(y_true, y_prob, k)

    metric_names = [f"{name}@{k}" for k in TOP_K_VALUES for name in ("Recall", "nDCG")]
    print("\nTest results")
    print(f"{'model':<12}" + "".join(f"{name:>11}" for name in metric_names))
    for model_name, scores in results.items():
        row = "".join(f"{scores[name]:>11.4f}" for name in metric_names)
        print(f"{model_name:<12}{row}")


if __name__ == "__main__":
    main()
