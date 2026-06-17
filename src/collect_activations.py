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
from typing import Optional, List, Tuple

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


def collect_activations(
    corpus_path: str,
    output_dir: str = "cache",
    layer: int = 8,
    max_sentences: int = 50_000,
    max_length: int = 64,
    batch_size: int = 16,
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

    # ── Collect activations ───────────────────────────────────────────────────
    all_acts: List[torch.Tensor] = []
    token_metadata: List[Tuple[str, int]] = []  # (token_str, sentence_idx)

    logger.info(f"Collecting activations from layer {layer}...")
    failed = 0

    for sent_idx, text in enumerate(tqdm(texts, desc="Processing sentences")):
        try:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():
                model(**tokens)

            act = get_activation(layer_key)  # (1, seq_len, 768)
            if act is None:
                continue

            act = act.squeeze(0).cpu()  # (seq_len, 768)

            if use_half:
                act = act.half()

            all_acts.append(act)

            # Store token strings for later interpretability lookups
            token_strings = tokenizer.convert_ids_to_tokens(
                tokens["input_ids"][0].cpu()
            )
            for tok_str in token_strings:
                token_metadata.append((tok_str, sent_idx))

        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Sentence {sent_idx} failed: {e}")
            continue

    remove_hooks()

    if not all_acts:
        raise RuntimeError(
            "No activations collected! Check your corpus path and format."
        )

    logger.info(f"Processed {len(all_acts):,} sentences ({failed} failed)")

    # ── Concatenate and save ──────────────────────────────────────────────────
    activation_tensor = torch.cat(all_acts, dim=0)  # (N_tokens, 768)

    acts_path = Path(output_dir) / f"layer{layer}_acts.pt"
    meta_path = Path(output_dir) / f"layer{layer}_token_metadata.pt"
    texts_path = Path(output_dir) / f"layer{layer}_source_texts.pt"

    torch.save(activation_tensor, acts_path)
    torch.save(token_metadata, meta_path)
    torch.save(texts, texts_path)

    # ── Stats report ──────────────────────────────────────────────────────────
    size_mb = acts_path.stat().st_size / 1e6
    logger.info(
        f"\n{'─'*50}\n"
        f"✅ ACTIVATION COLLECTION COMPLETE\n"
        f"   Shape:       {activation_tensor.shape}\n"
        f"   Dtype:       {activation_tensor.dtype}\n"
        f"   Value range: [{activation_tensor.float().min():.3f}, "
        f"{activation_tensor.float().max():.3f}]\n"
        f"   Tokens:      {activation_tensor.shape[0]:,}\n"
        f"   Saved to:    {acts_path} ({size_mb:.1f} MB)\n"
        f"{'─'*50}"
    )

    return activation_tensor, token_metadata


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
        "--batch-size", type=int, default=16,
        help="Sentences per batch"
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
        use_half=args.half,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
