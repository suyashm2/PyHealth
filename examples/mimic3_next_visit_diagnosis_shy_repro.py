"""Next-visit diagnosis prediction on MIMIC-III with SHy and baseline models.

Trains SHy (Yu et al., CHIL 2025) and the PyHealth RNN, RETAIN and Transformer
baselines on the NextVisitDiagnosisMIMIC3 task, and reports Recall@k and
nDCG@k, the metrics used by the SHy paper.

The pipeline follows PyHealth's five steps:
    1. Load the dataset.
    2. Set the task (one sample per patient: earlier visits predict the last).
    3. Split by patient into train, validation and test sets.
    4. Train each model with the PyHealth Trainer.
    5. Evaluate on the test set with Recall@k and nDCG@k.

By default the script uses PyHealth's public synthetic MIMIC-III data, so it
runs without credentials. To use the real MIMIC-III 1.4 data, pass the folder
that holds its CSV files:

    python examples/mimic3_next_visit_diagnosis_shy_repro.py \\
        --root /path/to/mimic-iii/1.4

Use --models to train only some of the models, for example --models shy.

Settings for SHy on real MIMIC-III (the default learning rate of 1e-3 is too
high for SHy there and makes training unstable):

    python examples/mimic3_next_visit_diagnosis_shy_repro.py \\
        --root /path/to/mimic-iii/1.4 --models shy \\
        --embedding-dim 128 --hidden-dim 128 --batch-size 64 \\
        --learning-rate 3e-4 --max-grad-norm 5 --epochs 100 --patience 15

Ablations (settings the SHy paper did not test):
    --augmentation-ratio 0   SHy without its false-negative augmentation,
                             which normally adds likely missing diagnoses
                             (default 0.2, the paper's setting).
    --prediction-activation sigmoid
                             Sigmoid instead of softmax for SHy's output
                             scores.
    --deterministic-eval-masks
                             Fixed phenotype masks at test time, instead of
                             random masks as in training (the authors' code).
    --sample-per-visit       One sample for every visit after the first,
                             instead of one sample per patient as in the paper.
    --alpha-loss-weight 0    SHy without its alpha regularizer, which keeps
                             the phenotype weights from collapsing onto one
                             phenotype (default 0.04, from the paper's table;
                             the authors' released run command uses 0).
    --embedding-init-std 0.02
                             Smaller starting weights for the code
                             embeddings (default 1.0, as in the authors'
                             code). On MIMIC-III this lowered Recall@20 by
                             about 2.9 points (.333 vs .362).

Learning-rate schedule (PyHealth's Trainer keeps the rate fixed):
    --lr-schedule plateau    Multiply the learning rate by --lr-decay-factor
                             whenever the validation score has not improved
                             for --lr-decay-patience epochs.

Example baseline output on the synthetic MIMIC-III data (2,085 samples, 20
epochs, seed 42, one NVIDIA RTX 4070 laptop GPU, about 2 minutes):

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
from pyhealth.models import SHyRepro
from pyhealth.tasks import NextVisitDiagnosisMIMIC3
from pyhealth.trainer import Trainer

SYNTHETIC_MIMIC3_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/"
MODEL_CLASSES = {"rnn": RNN, "retain": RETAIN, "transformer": Transformer, "shy": SHyRepro}
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


class PlateauTrainer(Trainer):
    """Trainer that lowers the learning rate when the validation score stalls.

    PyHealth's Trainer has no learning-rate schedule. This subclass builds the
    Adam optimizer itself (pass make_optimizer as optimizer_class) and steps a
    ReduceLROnPlateau scheduler after each validation pass.
    """

    def __init__(self, *args, val_loader, monitor, factor, patience, min_lr, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_loader = val_loader  # only validation passes step the schedule
        self.monitor = monitor  # validation score to watch, e.g. pr_auc_samples
        self.factor = factor  # multiply the learning rate by this when stuck
        self.patience = patience  # epochs without improvement before lowering
        self.min_lr = min_lr  # never go below this learning rate
        self.scheduler = None  # created together with the optimizer

    def make_optimizer(self, params, **optimizer_params):
        """Builds Adam and the scheduler that will adjust its learning rate."""
        optimizer = torch.optim.Adam(params, **optimizer_params)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )
        return optimizer

    def evaluate(self, dataloader):
        """Evaluates as usual, then updates the schedule on validation passes."""
        scores = super().evaluate(dataloader)
        if self.scheduler is not None and dataloader is self.val_loader:
            old_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            self.scheduler.step(scores[self.monitor])
            new_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"Learning rate lowered from {old_lr:.2e} to {new_lr:.2e}")
        return scores


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
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="stop after this many epochs without a better validation score",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="clip gradients to this norm (default: no clipping)",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "plateau"],
        default="none",
        help="plateau: lower the learning rate when validation stops improving",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="multiply the learning rate by this when it is lowered",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=3,
        help="epochs without improvement before the learning rate is lowered",
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-5, help="lowest learning rate allowed"
    )
    parser.add_argument("--sample-per-visit", action="store_true")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="SHy code embedding size (split evenly over the 4 ICD-9 levels)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="SHy GRU and phenotype size (default: same as --embedding-dim)",
    )
    parser.add_argument(
        "--augmentation-ratio",
        type=float,
        default=0.2,
        help="SHy false-negative augmentation ratio p; 0 turns it off",
    )
    parser.add_argument(
        "--prediction-activation",
        choices=["softmax", "sigmoid"],
        default="softmax",
        help="SHy output activation (the authors' code uses softmax)",
    )
    parser.add_argument(
        "--alpha-loss-weight",
        type=float,
        default=0.04,
        help="weight of SHy's alpha regularizer; 0 turns it off",
    )
    parser.add_argument(
        "--embedding-init-std",
        type=float,
        default=1.0,
        help="standard deviation of SHy's starting code-embedding weights",
    )
    parser.add_argument(
        "--deterministic-eval-masks",
        action="store_true",
        help="threshold SHy's phenotype masks at test time instead of sampling",
    )
    parser.add_argument("--dev", action="store_true", help="use a small subset")
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for model weights and batches"
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="seed for the train/validation/test split (keep fixed to compare runs)",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the five-step pipeline for each model."""
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
        sample_dataset, [0.7, 0.1, 0.2], seed=args.split_seed
    )
    train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, args.batch_size, shuffle=False)

    results = {}
    for model_name in args.models:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # STEP 4: train, keeping the epoch with the best validation PR-AUC
        if model_name == "shy":
            model = SHyRepro(
                dataset=sample_dataset,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                augmentation_ratio=args.augmentation_ratio,
                prediction_activation=args.prediction_activation,
                sample_phenotypes_at_eval=not args.deterministic_eval_masks,
                alpha_loss_weight=args.alpha_loss_weight,
                embedding_init_std=args.embedding_init_std,
            )
        else:
            model = MODEL_CLASSES[model_name](dataset=sample_dataset)
        if args.lr_schedule == "plateau":
            trainer = PlateauTrainer(
                model=model,
                metrics=["pr_auc_samples"],
                val_loader=val_loader,
                monitor="pr_auc_samples",
                factor=args.lr_decay_factor,
                patience=args.lr_decay_patience,
                min_lr=args.min_lr,
            )
            optimizer_class = trainer.make_optimizer
        else:
            trainer = Trainer(model=model, metrics=["pr_auc_samples"])
            optimizer_class = torch.optim.Adam
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_class=optimizer_class,
            optimizer_params={"lr": args.learning_rate},
            monitor="pr_auc_samples",
            patience=args.patience,
            max_grad_norm=args.max_grad_norm,
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
