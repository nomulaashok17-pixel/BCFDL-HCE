# Reproducibility protocol

## Configuration

All material hyperparameters are stored in `config.yaml`. Experiments record the resolved configuration, seed, software versions, device, client allocation and output hashes.

## Determinism

Python, NumPy and PyTorch seeds are set for each repetition. CUDA deterministic settings are enabled where supported. Some GPU kernels can remain hardware-dependent; therefore repeated-run statistics are reported instead of relying on a single run.

## Experimental design

- Primary end-to-end task: binary legitimate-versus-malicious ToN-IoT authentication/security decision.
- Subsystem benchmarks: MNIST and CIFAR-10.
- Default clients: 10.
- Default communication rounds: 100.
- Default local epochs: 5.
- Default batch size: 64.
- Default optimizer: Adam at 0.001.
- Default HCE length: 64 bits.
- Default client heterogeneity: Dirichlet non-IID with alpha 0.5.

## Required comparisons

The baseline script evaluates FedAvg, FedProx, SCAFFOLD-style control variates, FedNova normalization, MOON-style representation regularization and FedDyn-style dynamic regularization under common partitions and seeds.

## Required ablations

The component study evaluates:

1. FDL only.
2. FDL + AF-MFA.
3. FDL + HCE.
4. FDL + blockchain/IPFS integrity layer.
5. Full BCFDL-HCE.

A second ablation disables clustered odd/even client scheduling.

## Security evaluation

Attack code is intended for controlled reproducibility experiments against the local federated simulator. It does not provide network exploitation functionality. Attacks include label flipping, sign-flip poisoning, model replacement/scaling, random Byzantine updates, replay-event testing and Sybil-weight stress tests.

## Outputs

Each run writes machine-readable JSON/CSV artifacts and figures to the configured output directory. Reported paper values should be copied from these generated artifacts, never typed into the code.
