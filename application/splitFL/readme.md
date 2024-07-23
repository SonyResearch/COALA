# SplitFL

This application is a re-implementation of MocoSFL and vanilla supervised SFL based on COALA framework.

## Demo

**Run SFL:**

`
python application/splitFL/examples/cifar10_sfl.py
`

**Run MocoSFL Locally:**

`
python application/splitFL/examples/cifar10_mocosfl.py
`

**Run MocoSFL Remotely:**

In separate terminals:
`
python application/splitFL/examples/remote_server.py --is-remote True --local-port 22999
python application/splitFL/examples/remote_client.py --is-remote True --local-port 23400 --index 0
python application/splitFL/examples/remote_client.py --is-remote True --local-port 23401 --index 1
python application/splitFL/examples/remote_run.py --server-addr localhost:22999 --source manual
`

## Results


| Scheme| Arch | KNN/val-accu | linear eval | semi-10% | semi-1% | 
| -------------------- | ------------| ------- |------- |------- |------- |
| SFL (100 clients, 200 rounds)| ResNet18 (cut-1) |  94.79% | NA | NA | NA |
| MocoSFL (100 clients, 200 rounds)| ResNet18 (cut-1) |  77.15% | 79.14% | 78.31% | 73.09% |
| MocoSFL (100 clients, 400 rounds)| ResNet18 (cut-1) |  79.76% | 81.50% | 78.28% | 75.29% |
| MocoSFL (100 clients, 400 rounds)| ResNet18 (cut-2) |  80.29% | 81.14% | 78.50% | 75.37% |

## TODO
- [x] Implement Basic Functionality of SFL
- [x] Implement Basic Functionality of MocoSFL
- [x] Fully tested SFL
- [x] Implement Increased Sync Frequency of MocoSFL
- [x] Imeplement Cos Annearling, multi-step LR scheduleing
- [x] Fully tested MocoSFL
- [x] Clean up
- [x] Implement Remote Protocol
- [x] Implement ViT split Models

## Integration Guide

Change datasets and models in `application/splitFL/mocosfl_coordinator.py`



## Reference
1. Thapa, Chandra, et al. "Splitfed: When federated learning meets split learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.
2. Li, Jingtao, et al. "MocoSFL: enabling cross-client collaborative self-supervised learning." The Eleventh International Conference on Learning Representations. 2022.
