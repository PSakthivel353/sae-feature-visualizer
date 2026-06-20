# SAE Feature Visualizer

**An interactive mechanistic interpretability toolkit for discovering human-interpretable concepts inside GPT-2's hidden states, using Sparse Autoencoders with Top-K sparsity.**

![status](https://img.shields.io/badge/status-active-5EEAD4) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C) ![streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B) ![hardware](https://img.shields.io/badge/runs%20on-CPU%20%7C%20GPU-8B95A7)

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
15. [Project Evolution / Changelog](#15-project-evolution--changelog)

---

## 1. Project Overview

Large language models like GPT-2 produce fluent text, but the internal computations that produce that text are opaque — a 768-dimensional vector of floating-point numbers at each layer tells us almost nothing on its own. **SAE Feature Visualizer** is an end-to-end pipeline that:

1. Intercepts GPT-2's internal hidden states using PyTorch forward hooks
2. Trains a **Sparse Autoencoder (SAE)** — using **Top-K sparsity** — to decompose those hidden states into thousands of sparse, more-interpretable "features"
3. Empirically discovers what each feature represents by mining a large text corpus for its top-activating examples
4. Surfaces everything in a professional, interactive Streamlit dashboard

This project is a small-scale, from-scratch implementation of the same category of methodology used in published interpretability research — most notably Anthropic's work on decomposing transformer activations into monosemantic features via sparse dictionary learning. It is designed both as a working research tool and as a portfolio piece demonstrating genuine understanding of transformer internals, not just API usage.

The pipeline was built and iterated end-to-end on **consumer CPU-only hardware** — the dataset loader, activation cache format, and training configuration were all specifically hardened for that constraint (see [§15 Project Evolution](#15-project-evolution--changelog)).

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
Hidden layer (sparse)   ← only the top-K values survive; everything else is zeroed
        │
        ▼
     Linear                ← "decoder": projects back down
        │
        ▼
Reconstructed activation (768-dim)
```

**Why does this help?** Without the sparsity constraint, the wider hidden layer would just learn another entangled, superposed representation — the same problem, restated. Forcing the model to explain each input activation using only a *handful* of the available features pushes individual features to specialize: instead of "a bit of everything," each feature tends to fire cleanly for one concept (e.g., legal terminology, French loanwords, negation, code syntax).

### From L1 to Top-K: why the sparsity mechanism changed

The first working version of this project used a **plain L1 penalty** on the hidden activations:

```
Loss = MSE(reconstruction, original_activation) + λ · mean(|hidden_activations|)
```

This is the classical SAE formulation, and it works in principle — but in practice, `λ` only *encourages* sparsity indirectly. The number of active features per token is an emergent side-effect of the penalty weight, not something you control directly. Across training runs, the active-feature count would drift outside the target **20–50 active features per token** range and require constant `λ` re-tuning, with no guarantee of landing in range at all.

The project switched to **Top-K sparsity**, which enforces the constraint directly:

```
hidden = ReLU(encoder(x))
hidden = keep_top_k(hidden, k=32)   # zero out everything except the k largest activations
```

Instead of *hoping* a penalty weight produces the right sparsity level, Top-K **guarantees** exactly `k` active features per token, every time, by construction. This removed an entire axis of hyperparameter search and made the 20–50-active-feature target trivial to hit exactly (set `k` directly inside that range). The trade-off: Top-K is a harder constraint (no graceful degradation near the boundary), so `k` itself still needs sensible tuning — but tuning one integer is far simpler than tuning a continuous penalty weight indirectly.

The training objective is now:

```
Loss = MSE(reconstruction, original_activation)   # sparsity is enforced structurally, not via a loss term
```

Checkpoints now explicitly record `sparsity_mode: "topk"` and the `topk` value used, so any downstream script (analysis, dashboard) loads the correct decoding behavior automatically rather than assuming L1.

---

## 4. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            PHASE 1 — COLLECT                              │
│                                                                            │
│   Text Corpus (zstd-compressed)     GPT-2 (frozen, pretrained)           │
│   data/*.jsonl.zst            ──▶   12 transformer layers, d_model=768   │
│                                       │                                   │
│                                       │  forward hook on layer 8          │
│                                       ▼                                   │
│                            Hidden state vectors (per token, 768-dim)     │
│                                       │                                   │
│                                       ▼  (written in CHUNKS, not RAM-held)│
│                     cache/layer8_chunk_0000.pt, _0001.pt, ...            │
│                     cache/layer8_manifest.json  (chunk index)            │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            PHASE 2 — TRAIN                                │
│                                                                            │
│        768 ──Linear+ReLU──▶ hidden ──Top-K mask──▶ hidden ──Linear──▶ 768│
│        │                        │         │                  │          │
│    input act.              raw scores   exactly k          reconstruction│
│                                          survive per token                │
│                                                                            │
│        Loss = reconstruction_MSE only (sparsity enforced structurally)   │
│        Hidden size / batch size / epochs tuned down for CPU feasibility  │
│                                       │                                   │
│                                       ▼                                   │
│        checkpoints/sae_layer8.pt                                         │
│          { sparsity_mode: "topk", topk: <k>, ... }                       │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3 — ANALYZE & VISUALIZE                    │
│                                                                            │
│   For each feature:                                                      │
│     • Sweep entire corpus (single-file OR chunked cache — both supported)│
│     • Top-K-aware: zero activations are expected, not a bug              │
│     • Record top-N highest-activating tokens/sentences                   │
│     • Sanity test uses LAST-TOKEN sentence representation                │
│       (not a mean-pooled average — see §15 for why this changed)         │
│                                       │                                   │
│                                       ▼                                   │
│   cache/feature_top_examples.json   cache/feature_stats.json             │
│                                       │                                   │
│                                       ▼                                   │
│            ┌─────────────────────────────────────┐                       │
│            │   Streamlit Dashboard (dashboard.py)  │                      │
│            │  • Checkpoint / feature selector       │                      │
│            │  • Search · histograms · top examples  │                      │
│            │  • Live inference on typed sentences   │                      │
│            └─────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Installation Guide

### 5.1 Prerequisites

- Python 3.10 or later
- ~6 GB free disk space (corpus + cache + checkpoints)
- **CPU-only is fully supported** — the training configuration was specifically tuned for laptop-class hardware. A CUDA GPU speeds things up but is optional (see [§13 GPU Recommendations](#13-gpu-recommendations))

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

This includes `zstandard`, which is **required** — Pile and OpenWebText corpus shards are zstd-compressed, and `datasets` cannot decode them without it. Omitting this package is the single most common setup failure (see [§12 Troubleshooting](#12-troubleshooting)).

If you have an NVIDIA GPU and want CUDA acceleration, install the CUDA-enabled PyTorch build instead of the default:

```bash
# Example for CUDA 12.1 — check https://pytorch.org for your exact CUDA version
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps  # avoid reinstalling CPU torch
pip install transformers==4.40.1 datasets==2.19.1 streamlit==1.33.0 plotly==5.21.0 \
            numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 tqdm==4.66.4 pytest==8.2.0 zstandard==0.22.0
```

See [`requirements.txt`](./requirements.txt) for the full pinned dependency list.

---

## 6. Dataset Preparation

The pipeline expects a JSONL corpus, where each line is a JSON object with a `"text"` field. Corpus shards from the Pile or OpenWebText are **zstd-compressed**, so make sure `zstandard` is installed (§5.4) before downloading.

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

**Expected output after this stage:** your corpus file exists and `wc -l data/pile_sample.jsonl` reports your target sentence count.

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

### Chunked writing (memory-safe by design)

Activations are written to disk **incrementally, in chunks**, rather than accumulated in RAM and saved at the very end. The first working version of this script held every activation tensor in a Python list for the full run and only concatenated/saved at the end — this worked for small smoke-test corpora but reliably crashed with an out-of-memory error on full-size runs (50k+ sentences), because the entire multi-gigabyte tensor had to exist in memory simultaneously *in addition to* the model and intermediate buffers. Writing fixed-size chunks to disk as they fill, and tracking them via a manifest file, keeps peak memory bounded regardless of corpus size.

For memory-constrained machines, halve memory usage further with float16:
```bash
python src/collect_activations.py --half
```

### Expected output

```
cache/
├── layer8_chunk_0000.pt        # Activation chunk (chunk_size tokens, 768)
├── layer8_chunk_0001.pt
├── ...
├── layer8_manifest.json        # Index: chunk filenames, sizes, total token count
├── layer8_token_metadata.pt    # List of (token_string, sentence_index) pairs
└── layer8_source_texts.pt      # List of original sentences, for readable lookups
```

> **Note on cache format:** older single-file caches (`layer8_acts.pt`, one large tensor) remain fully supported. Training and analysis scripts detect which format is present — a manifest file means chunked, a single `.pt` tensor means legacy single-file — and load accordingly, so existing caches from earlier runs don't need to be regenerated.

Console output reports the final shape and value range once collection completes:
```
✅ ACTIVATION COLLECTION COMPLETE
   Total tokens: 1,487,032
   Chunks written: 24
   Value range:  [-8.214, 9.871]
   Manifest:     cache/layer8_manifest.json
```

**Sanity check:** values should roughly fall in `[-10, 10]`. Wildly larger ranges suggest a tokenization or hook misconfiguration.

---

## 8. SAE Training

```bash
python src/train_sae.py \
    --cache-path cache \
    --checkpoint-dir checkpoints \
    --d-hidden 1024 \
    --topk 32 \
    --epochs 10 \
    --batch-size 64 \
    --lr 2e-4
```

### CPU-tuned defaults

Hidden size, batch size, and epoch count were deliberately reduced from typical GPU-scale defaults (e.g. `d_hidden=4096`, `batch_size=256`) to make full training runs **feasible on a CPU-only laptop** in a reasonable amount of time, while still producing a hidden layer wide enough for individual features to specialize. If you do have GPU access, these can be scaled back up — see [§13 GPU Recommendations](#13-gpu-recommendations).

### Expected output

```
checkpoints/
├── sae_layer8.pt                # Final-epoch checkpoint
├── sae_layer8_best.pt           # Best validation-loss checkpoint
├── normalization_stats.pt       # Mean/std used to normalize activations (required at inference!)
└── training_history.json        # Per-epoch loss & sparsity curve, for plotting
```

Checkpoints now save explicit sparsity metadata:
```json
{
  "sparsity_mode": "topk",
  "topk": 32,
  "d_hidden": 1024,
  "epoch": 9,
  "val_loss": 0.0744
}
```

This lets `feature_analysis.py` and `dashboard.py` load any checkpoint and immediately know how to correctly interpret its hidden activations, instead of assuming a fixed sparsity mechanism.

Console output per epoch:
```
Epoch 7/10 │ Train Loss: 0.0842 │ Val Loss: 0.0851 │ Active Features: 32.0 ✅ (target: 20-50)
```

With Top-K, **Active Features is exactly `k` by construction** — it is no longer a diagnostic you need to chase by adjusting a penalty weight; it's a direct hyperparameter.

---

## 9. Feature Analysis

```bash
python src/feature_analysis.py \
    --checkpoint checkpoints/sae_layer8.pt \
    --cache-dir cache \
    --top-n 20 \
    --sanity-test
```

### What changed in the analysis logic

The analysis code originally assumed L1-style sparsity, where "inactive" was a fuzzy, near-zero threshold. Under Top-K, inactive features are **exactly** zero by construction, and the previous "near-zero" dead-feature detection logic needed correcting to treat exact zero as the explicit inactive signal rather than as a numerical edge case to be tolerant of. The feature-statistics step (frequency, mean/max activation, dead-feature flag) was updated accordingly.

The analysis pipeline also now transparently handles both cache formats from §7 (chunked manifest or single-file legacy tensor) when sweeping the corpus to build the feature catalogue.

### The sanity test: last-token vs. mean-pooled representation

The `--sanity-test` flag runs the project's core validation check: feed two semantically opposite sentences and verify their most active features are disjoint (no overlap = the SAE has learned meaningfully separated concepts, not noise).

```
sentence_a = "The attorney filed a motion in court"
sentence_b = "The protein binds to the receptor"
```

The original version of this test represented each sentence by **averaging the hidden-state activations across all its tokens**. This blurred the signal: GPT-2 is a left-to-right autoregressive model, so early tokens in a sentence carry far less contextual information than later ones, and naively averaging them dilutes whatever sharp, late-token signal the model has actually built up by the end of the sentence. The test was updated to instead use the **hidden state of the final token** — which has attended to the entire sentence and is the representation GPT-2 itself would use to predict what comes next — giving a much cleaner, more meaningful contrast between the two sentences' active features.

### Expected output

```
cache/
├── feature_top_examples.json    # {feature_id: [{token, sentence, score}, ...]}
├── feature_stats.json           # {feature_id: {frequency, mean_activation, max_activation, is_dead}}
└── sanity_test_result.json      # Pass/fail + diagnosis for the legal-vs-biology test
```

Console:
```
Dead features: 71/1024 (6.9%)
✅ SAE is working correctly — disjoint feature sets for unrelated concepts (last-token representation).
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

- **Checkpoint selector** — switch between trained SAE runs in the sidebar; each checkpoint's saved `sparsity_mode`/`topk` is read automatically
- **Feature Explorer** — browse all features, search by keyword, view activation histograms, top-examples tables, and per-feature statistics (all via Plotly)
- **Live Inference** — type any sentence, pick a token, and see which features fire in real time
- **About This Project** — in-app explanations of hidden states, SAEs, and feature discovery, with an ASCII pipeline diagram

---

## 11. Example Outputs

**Console output — training health check (epoch 10, Top-K=32):**
```
Epoch 10/10 │ Train Loss: 0.0731 │ Val Loss: 0.0744 │ Active Features: 32.0 ✅ (target: 20-50)
✅ TRAINING COMPLETE
   Final val loss:   0.0744
   Active feats:     32 / 1024 (fixed by Top-K)
   Sparsity mode:    topk (k=32)
```

**Feature interpretation example (illustrative — exact features depend on your corpus and seed):**

| Feature ID | Top activating tokens (illustrative) | Inferred concept |
|---|---|---|
| 88 | "motion", "attorney", "court", "plaintiff" | Legal/courtroom language |
| 204 | "not", "never", "no", "n't" | Negation |
| 391 | "le", "la", "de", "ne" | French-language tokens |
| 552 | "def", "return", "import", "==" | Code syntax |

**Sanity test output (last-token representation):**
```json
{
  "sentence_a": "The attorney filed a motion in court",
  "sentence_b": "The protein binds to the receptor",
  "top5_features_a": [88, 140, 233, 391, 502],
  "top5_features_b": [12, 97, 318, 444, 601],
  "overlap": [],
  "passed": true
}
```

**Dashboard view (described):** the Feature Explorer page shows four metric cards (total/live/dead features, average frequency) at the top, a search bar and feature-ID dropdown, then a two-tab panel — a teal Plotly histogram of activation strength on the left tab, and a sortable, progress-bar-annotated table of top examples on the right tab.

---

## 12. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `datasets` fails to load Pile/OpenWebText shards | Missing `zstandard` package | `pip install zstandard==0.22.0` — required for zstd-compressed corpus shards |
| `FileNotFoundError: data/pile_sample.jsonl` | Corpus not downloaded yet | Run `scripts/download_corpus.py` or check your `--corpus` path |
| `MemoryError` / process killed during activation collection | Old single-shot accumulation pattern, or chunk size too large | Use chunked writing (default) and reduce chunk size; don't hold the full corpus's activations in memory at once |
| Analysis script can't read an older cache | Cache format mismatch (chunked manifest vs. single-file) | Both formats are supported — confirm the manifest or single `.pt` file actually exists in `cache/` and matches the `--layer` you're pointing at |
| SAE loss never decreases / NaN loss | Activations not normalized | Confirm `train_sae.py` is normalizing (it does by default) — never feed raw activations directly |
| Active features not landing in 20–50 range (legacy L1 checkpoints) | L1 `λ` doesn't directly control sparsity | Prefer Top-K mode (`--topk 32`) for direct control; with L1, increase `λ` if too many features fire, decrease if too few |
| Training is slow / hangs on a laptop | Hidden size or batch size too large for CPU | Use the CPU-tuned defaults (`--d-hidden 1024 --batch-size 64`) rather than GPU-scale settings |
| Many features show `is_dead: true` (>40%) | Hidden dim too large relative to data, or `k` too small | Reduce `--d-hidden`, increase `--topk` slightly, or use a larger corpus |
| Dead-feature count looks wrong after switching to Top-K | Analysis still using old near-zero threshold logic | Confirm you're on the updated `feature_analysis.py`, which treats exact zero as inactive under Top-K rather than a fuzzy threshold |
| Sanity test shows overlapping features unexpectedly | Using mean-pooled sentence representation instead of last-token | Use the current sanity-test path, which represents each sentence by its last-token hidden state |
| Tokenizer padding error | GPT-2 has no pad token by default | Already handled in `hooks.py` via `tokenizer.pad_token = tokenizer.eos_token` |
| Streamlit shows "No checkpoints found" | Training hasn't run yet, or wrong directory | Run `train_sae.py` first; confirm `.pt` files exist in `checkpoints/` |
| Dashboard "Live Inference" activations look wrong | `normalization_stats.pt` missing, layer mismatch, or checkpoint missing `sparsity_mode` metadata | Re-run training to regenerate stats and metadata; ensure sidebar "Layer" matches the layer used in `collect_activations.py` |
| `ModuleNotFoundError: No module named 'sae_model'` | Running scripts from wrong working directory | Run all commands from the project root |
| Slow first run | Hugging Face is downloading GPT-2 weights (~500MB) | Normal — subsequent runs use the local cache (`~/.cache/huggingface`) |

---

## 13. GPU Recommendations

| Stage | CPU feasibility | Recommended GPU | Notes |
|---|---|---|---|
| Activation collection (50k sentences) | Workable with chunked writing (~30–60 min) | Any CUDA GPU with 4GB+ VRAM | GPT-2 base is small; even a laptop GPU helps a lot. Chunked writing keeps memory bounded on either device |
| SAE training (CPU-tuned: `d_hidden=1024`, `topk=32`) | Designed for this — runs comfortably on a laptop | 4GB+ VRAM lets you scale back up to `d_hidden=4096` | The SAE itself is tiny — bottleneck is data loading/IO, not model size |
| Feature analysis (sweep + top-N) | Fast even on CPU | Not required | Mostly a single forward pass over cached tensors |
| Dashboard | N/A (inference-only, on demand) | Not required | Live inference runs one sentence at a time — negligible compute |

**Optimization tips:**
- Use `--half` during activation collection to halve memory and disk usage; cast back to float32 (`acts.float()`) before training if you see precision issues.
- If you have GPU access, scale `d_hidden` back up (e.g. 4096) and increase `--batch-size` as far as VRAM allows — larger batches give more stable statistics per step, and Top-K's fixed-k behavior remains identical regardless of scale.
- Chunked activation writing is memory-safe on both CPU and GPU — keep it on even when you have plenty of VRAM, since the bottleneck during collection is host RAM, not GPU memory.
- If using a shared/cloud GPU (Colab, Kaggle, Lambda), checkpoint frequently — `train_sae.py` already saves both a best-validation checkpoint and a final checkpoint to guard against session timeouts.
- For larger GPT-2 variants (`gpt2-medium`/`large`/`xl`), VRAM requirements grow substantially — budget at least 8–16GB VRAM and proportionally larger `d_model` in your SAE config.

---

## 14. Future Extensions

This project is intentionally structured to grow into a more serious interpretability research tool. Natural next steps:

- **Multi-layer feature atlas** — train separate SAEs across all 12 GPT-2 layers and compare how features evolve through the network depth (early layers: syntax; later layers: semantics).
- **Feature steering / activation patching** — use discovered feature directions to *edit* model behavior at inference time (e.g., suppress a "toxicity" feature, amplify a "formality" feature) — a direct bridge into AI safety/alignment work.
- **Automated feature labeling via LLM** — replace the manual `label_features.py` workflow with an LLM call that reads each feature's top examples and proposes a label automatically, with human spot-checking.
- **Cross-model feature comparison** — train SAEs on multiple model families (GPT-2 vs. Pythia vs. a small LLaMA variant) and study which features are universal vs. model-specific.
- **Circuit-level analysis** — extend beyond single-feature interpretation to tracing how features in one layer causally influence features in the next (mirroring published circuits-style research).
- **Top-K vs. JumpReLU / batch-TopK variants** — explore newer sparsity mechanisms beyond plain Top-K (e.g. batch-wise Top-K, which controls sparsity per batch rather than per token) and compare resulting feature quality.
- **Scaling to GPU-scale dictionaries** — now that the cache and checkpoint formats are flexible, the same pipeline can be pointed at a GPU box and scaled to `d_hidden=16k+` without code changes.
- **Production hardening** — add experiment tracking (Weights & Biases), config management (Hydra), and a proper CI pipeline running the existing `pytest` suite on every commit.
- **Public deployment** — containerize the dashboard and deploy on Hugging Face Spaces or a similar platform for public, shareable feature exploration.

---

## 15. Project Evolution / Changelog

This project went through several rounds of real debugging and design correction. Documented here both as a project history and as evidence of the iterative engineering process behind the final pipeline.

| # | Change | Why it was needed |
|---|---|---|
| 1 | **Dataset loading + zstd fix** | The pipeline initially failed to load corpus files because Pile/OpenWebText shards require `zstandard` support, which wasn't installed. Adding the dependency fixed corpus loading. |
| 2 | **Activation collection: in-memory → chunked writing** | The original implementation held all activations in RAM and concatenated at the end; this crashed with out-of-memory errors on full-size corpora. Activations are now written to disk in chunks as they're produced, keeping peak memory bounded. |
| 3 | **Flexible cache format (single-file + chunked manifest)** | Needed so older single-tensor caches and the new chunked-cache layout can both be loaded interchangeably by training and analysis scripts, without forcing a re-collection of existing caches. |
| 4 | **Training tuned for laptop/CPU hardware** | Hidden size, batch size, and other training settings were reduced from GPU-scale defaults to make full training runs feasible on a CPU-only machine. |
| 5 | **Sparsity: plain L1 → Top-K** | L1 alone wasn't reliably keeping the active-feature count in the target 20–50 range — it only encourages sparsity indirectly via a penalty weight. Top-K was introduced to directly and exactly control how many features stay active per token. |
| 6 | **Checkpoint metadata now records sparsity config** | Saved checkpoints explicitly store `sparsity_mode: "topk"` and the `topk` value, so downstream analysis and dashboard code always know how to correctly interpret a given checkpoint's hidden activations. |
| 7 | **Feature analysis corrected for Top-K behavior** | Dead/inactive feature detection was updated to treat exact zero (Top-K's natural inactive state) correctly, and sentence-level representation choice for probing was revisited. |
| 8 | **Sanity test: mean-pooled → last-token representation** | Averaging all token activations blurred the sentence-level signal for an autoregressive model like GPT-2. Using the last token's hidden state — which has attended to the full sentence — gives a sharper, more meaningful contrast for the legal-vs-biology semantic test. |
| 9 | **Tests added/updated for new behavior** | Unit tests now cover the corpus loader, Top-K sparsity logic, and checkpoint metadata handling, so these don't silently regress in future changes. |

---

## Project Structure Reference

```
sae-feature-visualizer/
├── src/
│   ├── hooks.py                 # PyTorch forward hooks into GPT-2
│   ├── collect_activations.py   # Corpus → chunked activation cache + manifest
│   ├── sae_model.py             # SAE architecture (encoder/decoder, Top-K sparsity)
│   ├── train_sae.py             # Training loop, checkpointing (incl. sparsity metadata), CPU-tuned defaults
│   ├── feature_analysis.py      # Top-example mining, Top-K-aware stats, last-token sanity test
│   └── dashboard.py             # Streamlit interpretability dashboard
├── scripts/
│   ├── download_corpus.py       # Corpus download helper (Pile / OpenWebText, zstd-aware)
│   └── label_features.py        # Interactive CLI for manual feature labeling
├── tests/
│   ├── test_sae_model.py        # Unit tests for SAE architecture, loss & Top-K sparsity logic
│   └── test_hooks.py            # Unit tests for hook mechanism, corpus loader & checkpoint handling
├── data/                        # Corpus files (.jsonl) — gitignored
├── cache/                       # Activation chunks/manifest & feature analysis JSON — gitignored
├── checkpoints/                 # Trained SAE weights + sparsity metadata — gitignored
├── config.yaml                  # Central configuration defaults
├── requirements.txt             # Pinned dependency versions
└── README.md                    # This file
```

---

*Built as a hands-on exploration of mechanistic interpretability — understanding what's inside a language model, not just how to call one.*
