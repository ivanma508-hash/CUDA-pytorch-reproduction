# CUDA-PyTorch Reproduction

This repository contains a PyTorch reproduction of the GPU timing experiment in Table 4 of the paper.

## Main Notebook

- `CUDA_reproduction_final.ipynb`

The notebook reproduces the two-product value iteration setting and implements a batched PyTorch version of the GPU computation.

## Final Configuration

The final reported configuration is:

- GPU: Tesla T4
- dtype: `torch.float64`
- batch size: `256`
- iterations: `100`

This setting was selected because it provides the best tradeoff between numerical accuracy and runtime among the tested configurations.

## Main Result Summary

| Instance | Paper GPU Time | My Time | Final pi_est | Final Span |
|---|---:|---:|---:|---:|
| P1 | 91.29s | 89.41s | 4.503203516 | 2.84e-7 |
| P2 | 136.84s | 173.74s | 5.003599909 | 3.17e-7 |
| P3 | 214.71s | 340.33s | 5.508719865 | 3.49e-7 |
| P4 | 361.49s | 616.00s | 6.518342721 | 3.96e-7 |

The P1 runtime closely matches the paper's GPU runtime. P2-P4 are slower but remain in the same order of magnitude. The remaining gap is expected because this implementation uses batched PyTorch tensor operations rather than a hand-written CUDA kernel.

## Results

The `results/` folder contains timing logs for different combinations of:

- dtype: `float32` and `float64`
- batch size: `32`, `64`, `128`, and `256`

## Notes

The final implementation uses `torch.float64` with `batch_size=256` because it achieves stable convergence with final spans below `4e-7` for all P1-P4 instances. The faster `float32` versions are included in the results folder for comparison, but their final spans are around `1e-4` to `4e-4`.
