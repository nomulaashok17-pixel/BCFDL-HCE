# Dataset protocol

## Primary authentication dataset: ToN-IoT

BCFDL-HCE treats authentication intelligence as a binary decision task: **legitimate/benign** versus **malicious/suspicious** activity. The loader accepts a ToN-IoT CSV and performs the following leakage-safe pipeline:

1. Detect or use the configured target column.
2. Normalize the target into a binary label where benign/normal records map to 0 and attacks map to 1.
3. Drop obvious row identifiers and target aliases from predictors.
4. Replace infinities with missing values.
5. Fit categorical encoders and numerical scalers on the training partition only.
6. Split into train/validation/test sets with stratification.
7. Partition the training set across clients using IID or Dirichlet non-IID allocation.

The code intentionally does not fabricate a ToN-IoT file. If the CSV is absent, the ToN-IoT experiment exits with a clear instruction.

## MNIST

MNIST is used only to quantify federated convergence, communication volume and scheduling behavior on a simple benchmark. It is not evidence of MFA or blockchain effectiveness.

## CIFAR-10

CIFAR-10 is used only to stress the federated learning and HCE representation path using a more complex image benchmark. It is not evidence of end-to-end authentication effectiveness.

## Non-IID client generation

`federated.partition: dirichlet` draws class-wise client proportions from a Dirichlet distribution. `dirichlet_alpha` controls heterogeneity; smaller values produce stronger client drift. Allocation is deterministic for a fixed experiment seed.

## Data availability

The repository contains preprocessing and partitioning code but does not redistribute third-party datasets. Dataset access terms remain governed by their original providers.
