"""
collect_activations.py — Build the Activation Cache from a Text Corpus
─────────────────────────────────────────────────────────────────────────────
WHAT THIS DOES:
  1. Loads a large text corpus (JSONL format — one doc per line)
  2. Passes each sentence through GPT-2
  3. Captures the hidden-state activations at Layer 8 via forward hooks
  4. Saves all activation vectors to disk as a single .pt tensor

  After this script runs you'll have:
    cache/layer8_acts.pt         — shape (N_tokens, 768), dtype float32
    cache/layer8_token_texts.pt  — list of (token_string, sentence_index) pairs
                                   for mapping activations back to readable text

MEMORY NOTE:
  50k sentences × ~30 tokens × 768 dims × 4 bytes ≈ 4.4 GB (float32)
  Use --half flag to save as float16 (~2.2 GB). You can always cast back
  during training: acts.float()
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Any

# Import from sibling modules (add src/ to path when running directly)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from hooks import load_model_and_tokenizer, attach_hooks, get_activation, remove_hooks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str, max_sentences: int = 50_000) -> List[str]:
    """
    Load text from a JSONL file. Each line should be JSON with a 'text' key.

    Supports:
      - The Pile format:  {"text": "...", "meta": {...}}
      - OpenWebText:      {"text": "..."}
      - Plain text JSONL: {"text": "..."}

    Args:
        corpus_path:   Path to .jsonl corpus file
        max_sentences: Maximum number of sentences to load

    Returns:
        List of text strings
    """
    texts = []
    path = Path(corpus_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            f"Run scripts/download_corpus.py first, or see README § Dataset Preparation"
        )

    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("text", "").strip()
                if len(text) > 20:  # Skip very short lines
                    texts.append(text[:400])  # Cap length to avoid OOM
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(f"Loaded {len(texts):,} texts from {corpus_path}")
    return texts


def load_activation_cache(
    cache_path: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[Tuple[str, int]], List[str]]:
    """
    Load cached activations from either:
      - a single .pt tensor file, or
      - a manifest JSON produced by chunked collection.

    Returns:
        (acts, token_metadata, source_texts)
    """
    path = Path(cache_path)

    # If the user passes a missing default path, try to locate the chunk manifest.
    if not path.exists():
        base_name = path.stem.replace("_acts", "")
        candidate_paths = [
            path,
            path.parent / f"{base_name}_chunk_manifest.json",
            path.parent / f"{path.stem.replace('_acts', '')}_chunk_manifest.json",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                path = candidate
                break
        else:
            manifest_candidates = sorted(path.parent.glob("*_chunk_manifest.json"))
            if manifest_candidates:
                path = manifest_candidates[0]
            else:
                raise FileNotFoundError(
                    f"Could not find activation cache at '{cache_path}'. "
                    f"Expected either a merged tensor or a *_chunk_manifest.json file."
                )

    if path.suffix == ".json" and path.name.endswith("_chunk_manifest.json"):
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        chunk_files = manifest.get("chunks", [])
        chunk_tensors = []
        token_metadata = []

        for chunk in chunk_files:
            acts_path = path.parent / chunk["acts"]
            meta_path = path.parent / chunk["metadata"]
            if not acts_path.exists():
                raise FileNotFoundError(f"Missing chunk activation file: {acts_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing chunk metadata file: {meta_path}")
            chunk_tensors.append(torch.load(acts_path, map_location=device))
            token_metadata.extend(torch.load(meta_path, map_location="cpu"))

        acts = torch.cat(chunk_tensors, dim=0) if chunk_tensors else torch.empty(0, 0)
        source_texts_path = path.parent / f"{path.stem.replace('_chunk_manifest', '')}_source_texts.pt"
        source_texts = torch.load(source_texts_path, map_location="cpu") if source_texts_path.exists() else []
        return acts, token_metadata, source_texts

    # Support the legacy / default layout: layer8_acts.pt + layer8_token_metadata.pt
    acts = torch.load(path, map_location=device)
    base_name = path.stem.replace("_acts", "")
    token_metadata_path = path.parent / f"{base_name}_token_metadata.pt"
    source_texts_path = path.parent / f"{base_name}_source_texts.pt"

    token_metadata = torch.load(token_metadata_path, map_location="cpu") if token_metadata_path.exists() else []
    source_texts = torch.load(source_texts_path, map_location="cpu") if source_texts_path.exists() else []
    return acts, token_metadata, source_texts


def collect_activations(
    corpus_path: str,
    output_dir: str = "cache",
    layer: int = 8,
    max_sentences: int = 50_000,
    max_length: int = 64,
    batch_size: int = 16,
    chunk_size: int = 250_000,
    use_half: bool = False,
    model_name: str = "gpt2",
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, List[Tuple[str, int]]]:
    """
    Main function: run GPT-2 on corpus and collect activations.

    Args:
        corpus_path:   Path to JSONL corpus
        output_dir:    Where to save .pt files
        layer:         Which GPT-2 layer to hook (0–11)
        max_sentences: How many corpus sentences to process
        max_length:    Max token length per sentence (longer sequences truncated)
        batch_size:    Sentences per GPU batch (reduce if OOM)
        use_half:      Save as float16 to halve disk/memory usage
        model_name:    GPT-2 variant
        device:        'cuda' / 'cpu' / None (auto)

    Returns:
        (activation_tensor, token_metadata_list)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    attach_hooks(model, layers=[layer])
    layer_key = f"layer_{layer}"

    # ── Load corpus ───────────────────────────────────────────────────────────
    texts = load_corpus(corpus_path, max_sentences)

    # ── Collect activations in chunks ───────────────────────────────────────
    logger.info(f"Collecting activations from layer {layer}...")
    failed = 0
    total_tokens = 0
    chunk_idx = 0
    chunk_rows = 0
    chunk_acts: List[torch.Tensor] = []
    chunk_meta: List[Tuple[str, int]] = []
    chunk_manifest: List[Dict[str, str]] = []

    def flush_chunk() -> None:
        nonlocal chunk_idx, chunk_rows, total_tokens
        if not chunk_acts:
            return

        chunk_tensor = torch.cat(chunk_acts, dim=0)
        acts_path = Path(output_dir) / f"layer{layer}_acts_chunk_{chunk_idx:04d}.pt"
        meta_path = Path(output_dir) / f"layer{layer}_token_metadata_chunk_{chunk_idx:04d}.pt"

        torch.save(chunk_tensor, acts_path)
        torch.save(chunk_meta, meta_path)

        chunk_manifest.append(
            {
                "acts": acts_path.name,
                "metadata": meta_path.name,
                "rows": str(chunk_tensor.shape[0]),
            }
        )

        logger.info(
            f"Saved chunk {chunk_idx + 1}: {acts_path.name} "
            f"({chunk_tensor.shape[0]:,} rows)"
        )

        chunk_idx += 1
        chunk_rows = 0
        chunk_acts.clear()
        chunk_meta.clear()

    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_indices = range(batch_start, min(batch_start + batch_size, len(texts)))

        try:
            batch_tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            batch_tokens = {k: v.to(device) for k, v in batch_tokens.items()}

            with torch.no_grad():
                model(**batch_tokens)

            act = get_activation(layer_key)  # (batch, seq_len, 768)
            if act is None:
                continue

            act = act.detach().cpu()
            if use_half:
                act = act.half()

            input_ids = batch_tokens["input_ids"].cpu()
            attention_mask = batch_tokens.get("attention_mask", None)

            for i, sent_idx in enumerate(batch_indices):
                if attention_mask is not None:
                    valid_len = int(attention_mask[i].sum().item())
                else:
                    valid_len = int(input_ids[i].numel())

                if valid_len <= 0:
                    continue

                sample_act = act[i, :valid_len].contiguous()
                sample_ids = input_ids[i, :valid_len]

                if sample_act.shape[0] != sample_ids.shape[0]:
                    continue

                token_strings = tokenizer.convert_ids_to_tokens(sample_ids.tolist())

                chunk_acts.append(sample_act)
                for tok_str in token_strings:
                    chunk_meta.append((tok_str, sent_idx))

                chunk_rows += sample_act.shape[0]
                total_tokens += sample_act.shape[0]

                if chunk_rows >= chunk_size:
                    flush_chunk()

        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Batch starting at {batch_start} failed: {e}")
            continue

    flush_chunk()
    remove_hooks()

    if not chunk_manifest:
        raise RuntimeError(
            "No activations collected! Check your corpus path and format."
        )

    logger.info(
        f"Processed {len(texts):,} sentences ({failed} failed); "
        f"saved {total_tokens:,} token rows across {len(chunk_manifest)} chunks"
    )

    # ── Save metadata + manifest ─────────────────────────────────────────────
    texts_path = Path(output_dir) / f"layer{layer}_source_texts.pt"
    manifest_path = Path(output_dir) / f"layer{layer}_chunk_manifest.json"

    torch.save(texts, texts_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "layer": layer,
                "model_name": model_name,
                "chunk_size": chunk_size,
                "total_tokens": total_tokens,
                "chunks": chunk_manifest,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    logger.info(
        f"\n{'─'*50}\n"
        f"✅ ACTIVATION COLLECTION COMPLETE\n"
        f"   Tokens collected: {total_tokens:,}\n"
        f"   Chunks saved:     {len(chunk_manifest)}\n"
        f"   Cache dir:        {output_dir}\n"
        f"   Manifest:         {manifest_path}\n"
        f"{'─'*50}"
    )

    return torch.empty(0), []


def main():
    parser = argparse.ArgumentParser(
        description="Collect GPT-2 activations from a text corpus"
    )
    parser.add_argument(
        "--corpus", type=str, default="data/pile_sample.jsonl",
        help="Path to JSONL corpus file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="cache",
        help="Directory to save activation cache"
    )
    parser.add_argument(
        "--layer", type=int, default=8,
        help="GPT-2 layer index to hook (0–11)"
    )
    parser.add_argument(
        "--max-sentences", type=int, default=50_000,
        help="Maximum number of sentences to process"
    )
    parser.add_argument(
        "--max-length", type=int, default=64,
        help="Maximum token length per sentence"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Sentences per batch (lower for weaker systems)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Maximum token rows to keep in memory before flushing a chunk"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Save activations as float16 (saves ~50%% memory)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model variant"
    )

    args = parser.parse_args()

    collect_activations(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        layer=args.layer,
        max_sentences=args.max_sentences,
        max_length=args.max_length,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        use_half=args.half,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
