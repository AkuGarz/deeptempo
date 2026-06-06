# Roadmap

This is a rough plan. Dates are aspirational. Priorities shift based on what I need and what contributors show up for.

## v0.2 — Current (June 2026)

**Goal:** Stable single-GPU fine-tuning for PatchTST.

- [x] PatchTST model (base + tiny variants)
- [x] Fine-tuning loop (single GPU)
- [x] ETT dataset loaders (ETTh1, ETTh2, ETTm1, ETTm2)
- [x] Custom dataset class
- [x] Basic metrics (MSE, MAE, RMSE)
- [x] GPU memory helpers
- [ ] Proper LR scheduler with warmup (in progress)
- [ ] Gradient accumulation (in progress)
- [ ] Experiment tracking callbacks (wandb/MLflow)

## v0.3 — ~August 2026

**Goal:** Multi-GPU, LoRA, and TimesFM.

- [ ] Multi-GPU training (DDP)
- [ ] TimesFM adapter (stable, documented)
- [ ] LoRA fine-tuning support
- [ ] Hyperparameter config via YAML files
- [ ] Better checkpoint management (top-k, by metric)
- [ ] Pre-commit CI (linting, type checking)

## v0.4 — ~November 2026

**Goal:** More models, export, benchmarks.

- [ ] DLinear model
- [ ] iTransformer model (if I can get it working)
- [ ] ONNX export for inference
- [ ] Benchmark suite (6+ datasets, standardized eval)
- [ ] Quantization (int8 inference)
- [ ] Basic documentation site

## v1.0 — ~Q1 2027 (aspirational)

**Goal:** Mature toolkit.

- [ ] Model zoo with pretrained weights
- [ ] Zero-shot evaluation harness
- [ ] Community benchmark leaderboard
- [ ] Full documentation
- [ ] Stable API (no more breaking changes without deprecation)

## Backlog (not scheduled)

- [ ] Probabilistic forecasting (quantile outputs)
- [ ] Multivariate forecasting (proper channel mixing)
- [ ] Streaming/online fine-tuning
- [ ] ONNX Runtime integration for CPU inference
- [ ] Support for irregularly sampled time series
- [ ] Anomaly detection mode
