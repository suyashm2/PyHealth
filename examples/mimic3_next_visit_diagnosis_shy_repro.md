# SHyRepro on MIMIC-III next-visit diagnosis prediction: results

Results of `examples/mimic3_next_visit_diagnosis_shy_repro.py` for our
reproduction of SHy ([Yu et al., "Self-Explaining Hypergraph Neural Networks
for Diagnosis Prediction", CHIL 2025](https://proceedings.mlr.press/v287/yu25a.html))
and PyHealth's RNN, RETAIN and Transformer baselines.

## Setup

- **Data:** MIMIC-III v1.4, `DIAGNOSES_ICD` table. 7,499 patients have at
  least two visits (the paper reports 7,493). They use 4,350 distinct input
  diagnosis codes (the paper reports 4,880).
- **Task:** `NextVisitDiagnosisMIMIC3`, one example per patient: all visits
  except the last are the input, and the last visit's codes are the label.
- **Split:** 70/10/20 by patient (split seed 42). The checkpoint is chosen on
  the validation set. All scores below are on the test set.
- **SHy settings:** embedding and hidden size 128, learning rate 3e-4,
  gradient clipping at 5, up to 100 epochs with patience 15, and test
  predictions averaged over 10 random-mask passes (`--eval-passes 10`).
- **Hardware:** one NVIDIA RTX 4070 laptop GPU. A SHy run takes about 2
  hours; a RETAIN run takes about 27 minutes.

```
python examples/mimic3_next_visit_diagnosis_shy_repro.py \
    --root /path/to/mimic-iii/1.4 --models shy \
    --embedding-dim 128 --hidden-dim 128 --batch-size 64 \
    --learning-rate 3e-4 --max-grad-norm 5 --epochs 100 --patience 15 \
    --eval-passes 10
```

## Reproduction and baselines

Means over 3 seeds (42, 1, 2). The 3-seed SHy runs used batch size 8; batch
size 64 gives the same score (.3617 Recall@20, seed 42) in less time.

| Model | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 |
|---|---|---|---|---|
| SHy, paper (Table 1) | .2775 | .4135 | .3831 | .4088 |
| RETAIN, paper (Table 1) | .2024 | .3136 | .2958 | .3167 |
| SHyRepro | .2638 | .4080 | .3615 | .3970 |
| RNN, lr 1e-4 | .2581 | .4094 | .3509 | .3948 |
| Transformer, lr 1e-4 | .2526 | .3987 | .3447 | .3860 |
| RETAIN, lr 1e-4 | .2731 | .4332 | .3652 | .4135 |
| RETAIN, lr 3e-4, batch 8 (tuned) | .2871 | .4506 | .3816 | .4309 |

The baselines use batch size 8 and early stopping on validation. RETAIN was
then tuned on validation Recall@20 (see "How RETAIN was tuned" below), before
looking at any test score. The chosen settings are:

```
python examples/mimic3_next_visit_diagnosis_shy_repro.py \
    --root /path/to/mimic-iii/1.4 --models retain \
    --batch-size 8 --learning-rate 3e-4 --epochs 100 --patience 10
```

Findings:

- **SHyRepro comes within about 2 points of the paper.** Recall@20 is .3615
  against the paper's .3831. The three seeds range from .357 to .365.
- **Tuned RETAIN beats SHyRepro on all four metrics**, by 2.0 points of
  Recall@20, and trains about 4 times faster. Its three seeds range from
  .3785 to .3838, so the two ranges do not overlap.
- **Tuned RETAIN matches or beats the paper's SHy** on all four metrics. The
  paper reports RETAIN at .2958 Recall@20; ours reaches .3816 on our data.

## How RETAIN was tuned

All choices used validation Recall@20 only. The test patients were never used
for a choice. Every run used Adam, up to 100 epochs and early stopping with
patience 10. The tuning runs used a small separate script (not in this PR)
with the same data, task, split and RETAIN model as the example script.

**Step 1: learning rate and batch size.** One run each (seed 42), with
embedding size 128, dropout 0.5 and no weight decay.

| Learning rate | Batch 4 | Batch 8 | Batch 16 | Batch 32 |
|---|---|---|---|---|
| 1e-4 | .3804 | .3736 | .3621 | .3497 |
| 3e-4 | .3873 | **.3916** | .3826 | .3769 |
| 5e-4 | .3877 | .3908 | .3866 | .3850 |
| 7e-4 | .3885 | .3914 | **.3932** | .3869 |
| 1e-3 | .3881 | .3907 | .3905 | .3898 |

Batch size 64 was also tried at two learning rates and was lower (.3630 at
3e-4, .3781 at 1e-3).

**Step 2: change one other setting.** One run each (seed 42), starting from
lr 3e-4 at batch 8 (.3916). None of the changes helped.

| Change | Recall@20 |
|---|---|
| Embedding size 64 | .3769 |
| Embedding size 256 | .3823 |
| Dropout 0.3 | .3726 |
| Dropout 0.1 | .3361 |
| Weight decay 1e-5 | .3377 |

**3-seed check.** The two best cells of Step 1 are closer than the
run-to-run noise, so both were run on 3 seeds.

| Setting | Seed 42 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| lr 3e-4, batch 8 | .3916 | .3957 | .3955 | **.3943** |
| lr 7e-4, batch 16 | .3932 | .3907 | .3925 | .3921 |

**Chosen:** lr 3e-4, batch 8, embedding size 128, dropout 0.5, no weight
decay. Validation scores were used to choose the settings, so they run a
little high (.3943). The test score in the table above (.3816) is the fair
measure.

## Ablations (not in the paper)

One run each (seed 42, batch size 64), compared with the same run without
the change (.3617 Recall@20). Run-to-run noise is about 0.4 points.

| Change | Option | Recall@20 | Difference (points) |
|---|---|---|---|
| Sigmoid instead of softmax output | `--prediction-activation sigmoid` | .2485 | -11.3 |
| Fixed (thresholded) masks at test time | `--deterministic-eval-masks` | .2703 | -9.1 |
| Small starting weights, N(0, 0.02) | `--embedding-init-std 0.02` | .3329 (3 seeds) | -2.9 |
| False-negative augmentation off | `--augmentation-ratio 0` | .3596 | -0.2 |
| Alpha regularizer off | `--alpha-loss-weight 0` | .3596 | -0.2 |

- **The softmax output matters most.** Softmax makes all codes compete with
  each other; sigmoid scores each code on its own.
- **Random masks at test time help.** Averaging over sampled masks is much
  better than thresholding the mask probabilities once.
- **Augmentation and the alpha regularizer make no measurable difference**
  at these settings.

We also tried one example per visit (`--sample-per-visit`, 12,412 examples).
It scored .3625 Recall@20 in a single pass on its own test set, slightly
above comparable single-pass per-patient runs (about .354), but the test
examples differ, so the two are not directly comparable.

## Why we are still about 2 points below the paper

We compared our code with the authors' released code line by line and tested
each difference on MIMIC-III.

| Difference from the authors' code | Recall@20 | Effect (points) |
|---|---|---|
| Starting weights of the code embeddings: N(0, 0.02) changed to N(0, 1) | .333 to .362 | +2.9 |
| Wider message-passing layers (4 x 256), 3 seeds | .367 | +0.6, slower |
| Wider mask network (256) | .356 | none (within noise) |
| Linear temporal attention without bias | .364 | none (within noise) |
| Augmentation applied per phenotype | .358 | none (within noise) |
| Batch size 128, learning rate 1e-4, 450 epochs | .359 | none (within noise) |

Only the starting weights are part of this PR (`embedding_init_std`, default
1.0). The paper does not mention them. The other rows were tested in local
experiments and left out because they did not help or cost too much time.

Two things outside the model likely explain most of the remaining gap:

- **Preprocessing.** The paper has 4,880 input diagnosis codes and we get
  4,350 from the same database, so the preprocessing differs.
- **Checkpoint selection.** The authors' code has no validation set and picks
  the best epoch by test score. Doing the same adds about 0.3 points in our
  runs. We select on a validation set instead.
