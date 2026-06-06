# GPU Justification

## Why DeepTempo needs GPU resources

DeepTempo is a toolkit for **fine-tuning time-series foundation models**. Foundation models are transformer-based architectures that require significant compute during both fine-tuning and inference. Here's the breakdown.

## Fine-tuning ($CURRENT WORK)

Fine-tuning a PatchTST-base model (~1M params) on ETTh1 (1 year of hourly data):

| Hardware | Time per epoch | 10 epochs | Memory |
|---|---|---|---|
| CPU (AMD EPYC 7763) | ~12 min | ~120 min | RAM: 8 GB |
| NVIDIA A10G (24 GB) | ~45 sec | ~7.5 min | VRAM: 3.2 GB |
| NVIDIA T4 (16 GB) | ~90 sec | ~15 min | VRAM: 3.2 GB |
| NVIDIA A100 (80 GB) | ~22 sec | ~3.7 min | VRAM: 3.2 GB |

These numbers are from a single run, not averaged over trials. Your mileage will vary.

**The multiplier effect:** A hyperparameter sweep of 20 configurations × 10 epochs goes from 40 hours (CPU) to ~2.5 hours (A10G). That's the difference between iterating in a day vs. waiting a week.

## Planned: Larger models

As we add support for TimesFM (~200M params) and other foundation models, VRAM requirements will grow:

- PatchTST-base: ~1M params, ~3 GB VRAM (fine-tuning)
- TimesFM: ~200M params, ~12 GB VRAM (fine-tuning, estimated)
- TimesFM + LoRA: ~200M params, ~8 GB VRAM (estimated)
- Full fine-tuning of 300M+ param models: 20-30 GB VRAM (estimated)

## Planned: Multi-GPU training (v0.3)

Distributed Data Parallel (DDP) is on the roadmap. For larger models and datasets, multi-GPU training reduces wall-clock time. A 4×A10G setup would bring TimesFM fine-tuning down to ~10 min from ~45 min.

## Planned: Benchmark suite (v0.4)

We plan to build a standardized benchmark that evaluates fine-tuning performance across:
- 6+ datasets (ETT, Weather, Electricity, Traffic, etc.)
- 4+ model architectures
- Multiple forecast horizons

Running this benchmark once takes hours on GPU, days on CPU.

## Inference (future)

While inference is lighter than training, batched inference for evaluation still benefits from GPU:
- Evaluating on 100K+ samples with a 200M-param model takes ~30 min on GPU vs. hours on CPU.

## Summary

DeepTempo's core value is fast iteration on time-series fine-tuning. A GPU turns a "run overnight" workflow into a "run during coffee" workflow. This is especially important for independent researchers and practitioners who need to experiment quickly.

## Requested resources

- **For active development:** 1× A10G (24 GB) or equivalent — handles all current and near-future workloads
- **For benchmarking:** 4× A10G or 1× A100 — enables DDP and large-model experiments
- **Nice to have:** Cloud credits for CI/CD with GPU workers
