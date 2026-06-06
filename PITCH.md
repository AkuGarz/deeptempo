# DeepTempo — GPU Cloud Application Pitch

## What

DeepTempo is an open-source toolkit for fine-tuning small time-series foundation models (PatchTST, TimesFM, etc.) on domain-specific data. Think "HuggingFace transformers `Trainer` but for time-series forecasting."

## Why it matters

Time-series forecasting powers supply chain planning, energy grid management, demand prediction, and operational monitoring across every industry. Foundation models like TimesFM and PatchTST are making zero-shot forecasting viable — but most practitioners need to fine-tune these models on their specific data to get useful accuracy. There's no standard, lightweight toolkit for doing this.

DeepTempo fills that gap.

## Current state

- Early-stage (v0.2), open source (Apache 2.0)
- Working fine-tuning loop for PatchTST
- ETT dataset support
- ~200 GitHub stars (organic, no promotion)
- Active development (single maintainer, open to contributors)

## Why GPU resources are essential

Foundation model fine-tuning is compute-bound. A single hyperparameter sweep on CPU takes 40+ hours; on a single A10G it takes ~2.5 hours. As we add support for larger models (TimesFM: 200M params) and multi-GPU training, VRAM and compute requirements will increase. See `GPU_JUSTIFICATION.md` for detailed benchmarks.

## What we'd use GPU cloud resources for

1. **Development & CI:** GPU-enabled test suite for every PR (currently CPU-only CI, which misses GPU-specific bugs)
2. **Benchmarking:** Running the planned benchmark suite across 6+ datasets and 4+ models (hours of GPU time per run)
3. **Pretrained weight hosting:** Training and hosting baseline weights for the model registry
4. **Multi-GPU testing:** Validating DDP support before releasing v0.3

## Request

- Cloud GPU credits (any provider) — $500-2000 equivalent
- Or access to a single A10G/A100 instance for development
- 3-6 month duration

## Links

- GitHub: github.com/deeptempo/deeptempo
- License: Apache 2.0
- Maintainer: solo indie developer
