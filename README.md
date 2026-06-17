# SAE Feature Visualizer

**An interactive mechanistic interpretability toolkit for discovering human-interpretable concepts inside GPT-2's hidden states, using Sparse Autoencoders.**

![status](https://img.shields.io/badge/status-active-5EEAD4) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C) ![streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Is Mechanistic Interpretability?](#2-what-is-mechanistic-interpretability)
3. [What Is a Sparse Autoencoder?](#3-what-is-a-sparse-autoencoder)
4. [Architecture Diagram](#4-architecture-diagram)
5. [Installation Guide](#5-installation-guide)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Activation Collection](#7-activation-collection)
8. [SAE Training](#8-sae-training)
9. [Feature Analysis](#9-feature-analysis)
10. [Launching the Dashboard](#10-launching-the-dashboard)
11. [Example Outputs](#11-example-outputs)
12. [Troubleshooting](#12-troubleshooting)
13. [GPU Recommendations](#13-gpu-recommendations)
14. [Future Extensions](#14-future-extensions)

---

## 1. Project Overview

Large language models like GPT-2 produce fluent text, but the internal computations that produce that text are opaque — a 768-dimensional vector of floating-point numbers at each layer tells us almost nothing on its own. **SAE Feature Visualizer** is an end-to-end pipeline that:

1. Intercepts GPT-2's internal hidden states using PyTorch forward hooks
2. Trains a **Sparse Autoencoder (SAE)** to decompose those hidden states into thousands of sparse, more-interpretable "features"
3. Empirically discovers what each feature represents by mining a large text corpus for its top-activating examples
4. Surfaces everything in a professional, interactive Streamlit dashboard

This project is a small-scale, from-scratch implementation of the same category of methodology used in published interpretability research — most notably Anthropic's work on decomposing transformer activations into monosemantic features via sparse dictionary learning. It is designed both as a working research tool and as a portfolio piece demonstrating genuine understanding of transformer internals, not just API usage.

---

## 2. What Is Mechanistic Interpretability?

Mechanistic interpretability is the subfield of AI research focused on **reverse-engineering the internal computations of neural networks** into human-understandable algorithms and concepts — rather than treating models purely as black boxes evaluated only on their inputs and outputs.

**An analogy:** imagine a city (the neural network) with thousands of roads (neurons/activation dimensions). Traffic flows constantly, but you can't tell just by looking at any single road what kind of traffic it carries. Mechanistic interpretability is like installing a sensor system that watches the roads over time and learns: "Road 47 mostly carries food trucks delivering to downtown" — i.e., "Neuron/Feature 312 mostly activates when the model is processing legal language."

A central obstacle to this kind of analysis is the **superposition hypothesis**: neural networks often represent more distinct concepts than they have neurons or dimensions, by overlapping (superposing) multiple concepts onto the same dimensions. This means you cannot simply read off a neuron's meaning from its single most active examples — most neurons are entangled mixtures of several unrelated concepts. Sparse Autoencoders, described next, are the current leading technique for untangling this.

---

## 3. What Is a Sparse Autoencoder?

A **Sparse Autoencoder (SAE)** is a small neural network trained on a different objective than a normal autoencoder. Instead of compressing data into a *smaller* bottleneck, an SAE expands it into a much *larger* hidden layer — but constrains that hidden layer to be sparse (mostly zeros) for any given input.

```
Input activation (768-dim)
        │
        ▼
  Linear + ReLU            ← "encoder": projects up to a wider space
        │
        ▼
Hidden layer (4096-dim, SPARSE)   ← only ~20-50 of 4096 values are non-zero
        │
        ▼
     Linear                ← "decoder": projects back down
        │
        ▼
Reconstructed activation (768-dim)
```

**Why does this help?** Without the sparsity constraint, the wider hidden layer would just learn another entangled, superposed representation — the same problem, restated. The **L1 sparsity penalty** forces the model to explain each input activation using only a *handful* of the 4096 available features. Empirically, this pressure causes individual features to specialize: instead of "a bit of everything," each feature tends to fire cleanly for one concept (e.g., legal terminology, French loanwords, negation, code syntax).

The training objective combines two terms:

```
Loss = MSE(reconstruction, original_activation) + λ · mean(|hidden_activations|)
```

The reconstruction term keeps the SAE faithful to the original signal; the L1 term (weighted by `λ`, the most important hyperparameter in this project) pushes it toward sparse, interpretable solutions. Too high a `λ` and reconstruction degrades (too sparse to be useful); too low and features blur back into uninterpretable noise. Tuning `λ` to land at **20–50 active features per token** is the central calibration task of this project.

---

## 4. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            PHASE 1 — COLLECT                              │
│                                                                            │
│   Text Corpus              GPT-2 (frozen, pretrained)                    │
│   (50k+ sentences)   ──▶   12 transformer layers, d_model=768            │
│   data/*.jsonl              │                                            │
│                              │  forward hook on layer 8 (default)        │
│                              ▼                                            │
│                       Hidden state vectors (N_tokens × 768)              │
│                              │                                            │
│                              ▼                                            │
│                    cache/layer8_acts.pt  (activation cache)              │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            PHASE 2 — TRAIN                                │
│                                                                            │
│        768 ──Linear+ReLU──▶ 4096 (sparse) ──Linear──▶ 768                │
│        │                        │                       │                │
│    input act.            ~20-50 active              reconstruction       │
│                           features per token                             │
│                                                                            │
│        Loss = reconstruction_MSE + λ · L1(hidden)                        │
│                              │                                            │
│                              ▼                                            │
│              checkpoints/sae_layer8.pt  (trained weights)                │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3 — ANALYZE & VISUALIZE                    │
│                                                                            │
│   For each of 4096 features:                                             │
│     • Sweep entire corpus through SAE encoder                            │
│     • Record top-N highest-activating tokens/sentences                   │
│     • Compute frequency, mean/max activation, dead-feature flags         │
│                              │                                            │
│                              ▼                                            │
│   cache/feature_top_examples.json   cache/feature_stats.json             │
│                              │                                            │
│                              ▼                                            │
│            ┌─────────────────────────────────────┐                       │
│            │   Streamlit Dashboard (dashboard.py)  │                       │
│            │  • Checkpoint / feature selector       │                       │
│            │  • Search · histograms · top examples  │                       │
│            │  • Live inference on typed sentences   │                       │
│            └─────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Installation Guide

### 5.1 Prerequisites

- Python 3.10 or later
- ~6 GB free disk space (corpus + cache + checkpoints)
- A CUDA-capable GPU is strongly recommended but not required (see [§13 GPU Recommendations](#13-gpu-recommendations))

### 5.2 Folder Creation

If starting from scratch, recreate the full project structure with:

```bash
mkdir -p sae-feature-visualizer/src
mkdir -p sae-feature-visualizer/data
mkdir -p sae-feature-visualizer/checkpoints
mkdir -p sae-feature-visualizer/cache
mkdir -p sae-feature-visualizer/scripts
mkdir -p sae-feature-visualizer/tests
mkdir -p sae-feature-visualizer/docs
cd sae-feature-visualizer
```

### 5.3 Virtual Environment Setup

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Using conda (alternative):**
```bash
conda create -n sae-viz python=3.10 -y
conda activate sae-viz
```

### 5.4 Package Installation

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install the CUDA-enabled PyTorch build instead of the default:

```bash
# Example for CUDA 12.1 — check https://pytorch.org for your exact CUDA version
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps  # avoid reinstalling CPU torch
pip install transformers==4.40.1 datasets==2.19.1 streamlit==1.33.0 plotly==5.21.0 \
            numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 tqdm==4.66.4 pytest==8.2.0
```

See [`requirements.txt`](./requirements.txt) for the full pinned dependency list.

---

## 6. Dataset Preparation

The pipeline expects a JSONL corpus at `data/pile_sample.jsonl`, where each line is a JSON object with a `"text"` field.

### Option A — Automated download (recommended)

```bash
python scripts/download_corpus.py --source pile --n-samples 50000 --output data/pile_sample.jsonl
```

This streams from `monology/pile-uncopyrighted` on Hugging Face and writes 50,000 documents (~400–600MB depending on document length).

To use OpenWebText instead:
```bash
python scripts/download_corpus.py --source openwebtext --n-samples 50000 --output data/pile_sample.jsonl
```

### Option B — Manual / custom corpus

Any JSONL file with a `text` field works:
```json
{"text": "The lawyer argued brilliantly in front of the judge."}
{"text": "The protein binds to the receptor with high affinity."}
```

### Recommended corpus size

| Use case | Sentences | Approx. disk |
|---|---|---|
| Quick smoke test | 1,000–5,000 | ~20–80 MB |
| Meaningful feature discovery | 50,000 | ~400–600 MB |
| Higher-fidelity research run | 200,000+ | 2 GB+ |

**Expected output after this stage:** `data/pile_sample.jsonl` exists and `wc -l data/pile_sample.jsonl` reports your target sentence count.

---

## 7. Activation Collection

This stage runs GPT-2 over your corpus and caches hidden-state activations from a chosen layer.

```bash
python src/collect_activations.py \
    --corpus data/pile_sample.jsonl \
    --output-dir cache \
    --layer 8 \
    --max-sentences 50000 \
    --max-length 64
```

For memory-constrained machines, halve memory usage with float16:
```bash
python src/collect_activations.py --half
```

### Expected output

```
cache/
├── layer8_acts.pt              # Tensor (N_tokens, 768) — the activation cache
├── layer8_token_metadata.pt    # List of (token_string, sentence_index) pairs
└── layer8_source_texts.pt      # List of original sentences, for readable lookups
```

Console output reports the final shape and value range:
```
✅ ACTIVATION COLLECTION COMPLETE
   Shape:       torch.Size([1487032, 768])
   Dtype:       torch.float32
   Value range: [-8.214, 9.871]
   Tokens:      1,487,032
   Saved to:    cache/layer8_acts.pt (4563.2 MB)
```

**Sanity check:** shape should be `(N, 768)` and values should roughly fall in `[-10, 10]`. Wildly larger ranges suggest a tokenization or hook misconfiguration.

---

## 8. SAE Training

```bash
python src/train_sae.py \
    --cache-path cache/layer8_acts.pt \
    --checkpoint-dir checkpoints \
    --d-hidden 4096 \
    --epochs 10 \
    --batch-size 256 \
    --lr 2e-4 \
    --lambda-l1 1e-3
```

### Expected output

```
checkpoints/
├── sae_layer8.pt                # Final-epoch checkpoint
├── sae_layer8_best.pt           # Best validation-loss checkpoint
├── normalization_stats.pt       # Mean/std used to normalize activations (required at inference!)
└── training_history.json        # Per-epoch loss & sparsity curve, for plotting
```

Console output per epoch:
```
Epoch 7/10 │ Train Loss: 0.0842 │ Val Loss: 0.0851 │ Active Features: 34.2 ✅ (target: 20-50)
```

The **Active Features** metric is your primary health signal. The `✅`/`⚠️` flag tells you immediately whether `λ` needs adjustment — see [§12 Troubleshooting](#12-troubleshooting) for tuning guidance.

---

## 9. Feature Analysis

```bash
python src/feature_analysis.py \
    --checkpoint checkpoints/sae_layer8.pt \
    --cache-dir cache \
    --top-n 20 \
    --sanity-test
```

The `--sanity-test` flag runs the **semantic sanity test**: feeding two semantically opposite sentences ("The attorney filed a motion in court" vs. "The protein binds to the receptor") and verifying their top-5 activated features are disjoint.

### Expected output

```
cache/
├── feature_top_examples.json    # {feature_id: [{token, sentence, score}, ...]}
├── feature_stats.json           # {feature_id: {frequency, mean_activation, max_activation, is_dead}}
└── sanity_test_result.json      # Pass/fail + diagnosis for the legal-vs-biology test
```

Console:
```
Dead features: 312/4096 (7.6%)
✅ SAE is working correctly — disjoint feature sets for unrelated concepts.
```

A healthy run typically has **5–20% dead features** (never activate on your corpus) — this is normal for an overcomplete dictionary, not a bug.

(Optional) Manually label your most informative features for the dashboard:
```bash
python scripts/label_features.py --n-candidates 30
```

---

## 10. Launching the Dashboard

```bash
streamlit run src/dashboard.py
```

Open the printed local URL (typically `http://localhost:8501`). The dashboard provides:

- **Checkpoint selector** — switch between trained SAE runs in the sidebar
- **Feature Explorer** — browse all features, search by keyword, view activation histograms, top-examples tables, and per-feature statistics (all via Plotly)
- **Live Inference** — type any sentence, pick a token, and see which features fire in real time
- **About This Project** — in-app explanations of hidden states, SAEs, and feature discovery, with an ASCII pipeline diagram

---

## 11. Example Outputs

**Console output — training health check (epoch 10):**
```
Epoch 10/10 │ Train Loss: 0.0731 │ Val Loss: 0.0744 │ Active Features: 28.6 ✅ (target: 20-50)
✅ TRAINING COMPLETE
   Final val loss:   0.0744
   Avg active feats: 28.6 / 4096
   Diagnosis: ✅ Healthy sparsity range.
```

**Feature interpretation example (illustrative — exact features depend on your corpus and seed):**

| Feature ID | Top activating tokens (illustrative) | Inferred concept |
|---|---|---|
| 312 | "motion", "attorney", "court", "plaintiff" | Legal/courtroom language |
| 847 | "not", "never", "no", "n't" | Negation |
| 1502 | "le", "la", "de", "ne" | French-language tokens |
| 2210 | "def", "return", "import", "==" | Code syntax |

**Dashboard view (described):** the Feature Explorer page shows four metric cards (total/live/dead features, average frequency) at the top, a search bar and feature-ID dropdown, then a two-tab panel — a teal Plotly histogram of activation strength on the left tab, and a sortable, progress-bar-annotated table of top examples on the right tab.

---

## 12. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: cache/pile_sample.jsonl` | Corpus not downloaded yet | Run `scripts/download_corpus.py` or check your `--corpus` path |
| SAE loss never decreases / NaN loss | Activations not normalized | Confirm `train_sae.py` is normalizing (it does by default) — never feed raw activations directly |
| Active features stuck at 700+ | `λ` too low | Increase `--lambda-l1` (try 5e-3, then 1e-2) |
| Active features stuck near 0, recon loss high (>0.2) | `λ` too high | Decrease `--lambda-l1` (try 5e-4, then 1e-4) |
| `CUDA out of memory` during collection | Batch/sequence too large for GPU | Lower `--max-length`, use `--half`, or run on CPU |
| `CUDA out of memory` during training | Batch size too large | Lower `--batch-size` (e.g. 128 or 64) |
| Many features show `is_dead: true` (>40%) | Hidden dim too large relative to data, or λ too high | Reduce `--d-hidden` or lower `λ`; also try a larger corpus |
| Tokenizer padding error | GPT-2 has no pad token by default | Already handled in `hooks.py` via `tokenizer.pad_token = tokenizer.eos_token` — confirm you're using the provided loader |
| Streamlit shows "No checkpoints found" | Training hasn't run yet, or wrong directory | Run `train_sae.py` first; confirm `.pt` files exist in `checkpoints/` |
| Dashboard "Live Inference" activations look wrong | `normalization_stats.pt` missing or layer mismatch | Re-run training to regenerate stats; ensure sidebar "Layer" matches the layer used in `collect_activations.py` |
| `ModuleNotFoundError: No module named 'sae_model'` | Running scripts from wrong working directory | Run all commands from the project root, or rely on the `sys.path.insert` already present in each script |
| Slow first run | Hugging Face is downloading GPT-2 weights (~500MB) | Normal — subsequent runs use the local cache (`~/.cache/huggingface`) |

---

## 13. GPU Recommendations

| Stage | CPU feasibility | Recommended GPU | Notes |
|---|---|---|---|
| Activation collection (50k sentences) | Slow but workable (~30-60 min) | Any CUDA GPU with 4GB+ VRAM | GPT-2 base is small; even a laptop GPU helps a lot |
| SAE training (4096 hidden, 10 epochs) | Workable (~10-20 min on 1.5M tokens) | 4GB+ VRAM | The SAE itself is tiny (~6M params) — bottleneck is data loading, not model size |
| Feature analysis (sweep + top-N) | Fast even on CPU | Not required | Mostly a single forward pass over cached tensors |
| Dashboard | N/A (inference-only, on demand) | Not required | Live inference runs one sentence at a time — negligible compute |

**Optimization tips:**
- Use `--half` during activation collection to halve memory and disk usage; cast back to float32 (`acts.float()`) before training if you see precision issues.
- Increase `--batch-size` during training as far as your VRAM allows — larger batches give more stable sparsity statistics per step.
- If using a shared/cloud GPU (Colab, Kaggle, Lambda), checkpoint frequently — `train_sae.py` already saves both a best-validation checkpoint and a final checkpoint to guard against session timeouts.
- Pin `torch.backends.cudnn.benchmark = True` if processing many same-length batches (optional addition for advanced users).
- For larger GPT-2 variants (`gpt2-medium`/`large`/`xl`), VRAM requirements grow substantially — budget at least 8–16GB VRAM and proportionally larger `d_model` in your SAE config.

---

## 14. Future Extensions

This project is intentionally structured to grow into a more serious interpretability research tool. Natural next steps:

- **Multi-layer feature atlas** — train separate SAEs across all 12 GPT-2 layers and compare how features evolve through the network depth (early layers: syntax; later layers: semantics).
- **Feature steering / activation patching** — use discovered feature directions to *edit* model behavior at inference time (e.g., suppress a "toxicity" feature, amplify a "formality" feature) — a direct bridge into AI safety/alignment work.
- **Automated feature labeling via LLM** — replace the manual `label_features.py` workflow with an LLM call that reads each feature's top examples and proposes a label automatically, with human spot-checking.
- **Cross-model feature comparison** — train SAEs on multiple model families (GPT-2 vs. Pythia vs. a small LLaMA variant) and study which features are universal vs. model-specific.
- **Circuit-level analysis** — extend beyond single-feature interpretation to tracing how features in one layer causally influence features in the next (mirroring published circuits-style research).
- **Scaling laws for sparsity** — systematically sweep `d_hidden` and `λ` together and plot the reconstruction-vs-sparsity Pareto frontier, rather than hand-tuning a single value.
- **Production hardening** — add experiment tracking (Weights & Biases), config management (Hydra), and a proper CI pipeline running the existing `pytest` suite on every commit.
- **Public deployment** — containerize the dashboard and deploy on Hugging Face Spaces or a similar platform for public, shareable feature exploration.

---

## Project Structure Reference

```
sae-feature-visualizer/
├── src/
│   ├── hooks.py                 # PyTorch forward hooks into GPT-2
│   ├── collect_activations.py   # Corpus → activation cache
│   ├── sae_model.py             # SAE architecture (encoder/decoder + loss)
│   ├── train_sae.py             # Training loop, checkpointing, health diagnostics
│   ├── feature_analysis.py      # Top-example mining, stats, sanity test
│   └── dashboard.py             # Streamlit interpretability dashboard
├── scripts/
│   ├── download_corpus.py       # Corpus download helper (Pile / OpenWebText)
│   └── label_features.py        # Interactive CLI for manual feature labeling
├── tests/
│   ├── test_sae_model.py        # Unit tests for SAE architecture & loss
│   └── test_hooks.py            # Unit tests for hook mechanism
├── data/                        # Corpus files (.jsonl) — gitignored
├── cache/                       # Activation tensors & feature analysis JSON — gitignored
├── checkpoints/                 # Trained SAE weights — gitignored
├── config.yaml                  # Central configuration defaults
├── requirements.txt             # Pinned dependency versions
└── README.md                    # This file
```

---

*Built as a hands-on exploration of mechanistic interpretability — understanding what's inside a language model, not just how to call one.*
