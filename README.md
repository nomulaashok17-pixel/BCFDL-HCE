# BCFDL-HCE

Reproducibility code for **BCFDL-HCE: Blockchain-assisted Clustered Federated Deep Learning with Hash Context Encoding for decentralized IoT authentication**.

The repository implements one consistent experimental pipeline:

`authentication context -> local FDL model -> legitimacy/risk score -> HCE -> AF-MFA -> secure federated update -> Limited-Raft ledger validation -> content-addressed/IPFS storage -> authentication decision`

## Scope of the datasets

- **ToN-IoT** is the primary dataset for the authentication/security experiment. It is converted into a binary legitimate-versus-malicious decision task.
- **MNIST** is used only as a standardized federated-learning convergence and communication benchmark.
- **CIFAR-10** is used only as a standardized federated-learning/HCE scalability benchmark.

Results on MNIST or CIFAR-10 must not be interpreted as direct validation of the complete MFA/blockchain authentication framework.

## Implemented components

- Cluster-aware federated learning with reproducible odd/even cluster scheduling.
- Federated baselines: FedAvg, FedProx, SCAFFOLD-style control variates, FedNova normalization, MOON-style contrastive regularization, and FedDyn-style dynamic regularization.
- Hash Context Encoding (HCE) with supervised similarity, quantization and prototype-alignment terms.
- Adaptive Federated Multi-Factor Authentication (**AF-MFA**) with a genetic factor-selection policy.
- Gradient clipping, optional Gaussian DP perturbation and pairwise-mask secure aggregation.
- Permissioned append-only ledger with a **Limited-Raft** experimental coordinator; no PBFT code is used.
- SHA-256 model integrity verification and local content-addressed storage, with an optional IPFS HTTP adapter.
- Controlled replay, label-flip, sign-flip, scaling/model-replacement, Byzantine-random and Sybil-style attack experiments.
- Classification, communication, convergence, authentication-latency, ledger-latency, storage and security metrics.
- Five-way component ablation plus cluster-training ablation.
- Repeated-run statistical summaries with confidence intervals and paired significance tests.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## ToN-IoT data

Place a ToN-IoT CSV file at the path configured under `datasets.toniot_csv` in `config.yaml`. The loader searches common target columns such as `label`, `type`, `attack`, and `class`. See `DATASETS.md` for details.

## Quick smoke test

```bash
python smoke_test.py
```

This validates HCE, AF-MFA, secure aggregation, ledger integrity, non-IID partitioning, and a tiny federated training cycle without downloading external datasets.

## Main experiments

```bash
python experiments/toniot_authentication.py --config config.yaml
python experiments/fl_benchmarks.py --config config.yaml --dataset mnist
python experiments/fl_benchmarks.py --config config.yaml --dataset cifar10
python experiments/baseline_comparison.py --config config.yaml
python experiments/ablation.py --config config.yaml
python experiments/attack_robustness.py --config config.yaml
python experiments/privacy_analysis.py --config config.yaml
python experiments/system_overhead.py --config config.yaml
python experiments/statistical_analysis.py --config config.yaml
```

Run the configured suite with:

```bash
python reproduce.py --config config.yaml
```

All generated outputs are written under `outputs/` at run time. No manuscript performance number is hard-coded; every metric is computed from predictions or measured execution events.

## Reproducibility principles

1. One framework name: **BCFDL-HCE**.
2. One adaptive MFA name: **AF-MFA**.
3. One consensus assumption: **Limited-Raft** for the permissioned experimental ledger.
4. ToN-IoT is the primary end-to-end authentication dataset.
5. MNIST/CIFAR-10 are subsystem benchmarks only.
6. Every experiment is seeded and configuration-driven.
7. The repository separates measured results from analytical/security claims.

See `REPRODUCIBILITY.md` and `CODE_AVAILABILITY.md` before archival release.
