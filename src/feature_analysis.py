"""
feature_analysis.py — Discover What Each SAE Feature Represents
─────────────────────────────────────────────────────────────────────────────
WHAT THIS DOES:
  For every feature in the trained SAE (up to 4096 of them), find the
  top-N tokens/sentences from the corpus that activate it most strongly.
  Reading these top examples reveals the human-interpretable concept the
  feature has learned to detect (e.g., "legal language", "negation",
  "French words", "code syntax").

  Also runs the "Semantic Sanity Test" described in the project's testing
  strategy: feeds two semantically opposite sentences and checks whether
  they activate disjoint feature sets.

OUTPUT:
  cache/feature_top_examples.json  — {feature_id: [{text, token, score}, ...]}
  cache/feature_stats.json         — {feature_id: {frequency, mean_activation, ...}}
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from sae_model import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_trained_sae(checkpoint_path: str, device: str = "cpu") -> SparseAutoencoder:
    """Load a trained SAE from checkpoint, handling both old and new save formats."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        d_model = ckpt.get("d_model", 768)
        d_hidden = ckpt.get("d_hidden", 4096)
        sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden)
        sae.load_state_dict(ckpt["model_state_dict"])
    else:
        # Legacy format: raw state_dict
        sae = SparseAutoencoder()
        sae.load_state_dict(ckpt)

    sae.eval()
    sae = sae.to(device)
    return sae


def compute_feature_activations(
    sae: SparseAutoencoder,
    acts: torch.Tensor,
    norm_stats_path: Optional[str] = None,
    batch_size: int = 1024,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run all cached activations through the SAE encoder to get feature
    activations for every token in the corpus.

    Returns:
        hidden: (N_tokens, d_hidden) — sparse feature activation matrix
    """
    if norm_stats_path and Path(norm_stats_path).exists():
        stats = torch.load(norm_stats_path)
        acts = (acts - stats["mean"]) / (stats["std"] + 1e-8)
    else:
        logger.warning("No normalization stats found — using raw activations. "
                        "This may not match training distribution.")

    all_hidden = []
    sae.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(acts), batch_size), desc="Encoding tokens"):
            batch = acts[i:i + batch_size].to(device).float()
            hidden = sae.encode(batch)
            all_hidden.append(hidden.cpu())

    return torch.cat(all_hidden, dim=0)


def find_top_examples_per_feature(
    hidden: torch.Tensor,
    token_metadata: List,
    source_texts: List[str],
    top_n: int = 20,
    feature_ids: Optional[List[int]] = None,
) -> Dict[int, List[Dict]]:
    """
    For each feature, find the top-N tokens that activate it most strongly.

    Args:
        hidden:          (N_tokens, d_hidden) feature activation matrix
        token_metadata:  list of (token_str, sentence_idx) per row of `hidden`
        source_texts:    list of original sentences, indexed by sentence_idx
        top_n:           how many top examples to keep per feature
        feature_ids:     specific features to analyze (default: all)

    Returns:
        {feature_id: [{"token": str, "sentence": str, "score": float}, ...]}
    """
    n_features = hidden.shape[1]
    if feature_ids is None:
        feature_ids = list(range(n_features))

    results = {}
    for fid in tqdm(feature_ids, desc="Finding top examples"):
        scores = hidden[:, fid]
        if scores.max() == 0:
            results[fid] = []  # Dead feature — never activates
            continue

        k = min(top_n, (scores > 0).sum().item())
        if k == 0:
            results[fid] = []
            continue

        top_vals, top_idxs = scores.topk(k)
        examples = []
        for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            tok_str, sent_idx = token_metadata[idx]
            sentence = source_texts[sent_idx] if sent_idx < len(source_texts) else ""
            examples.append({
                "token": tok_str,
                "sentence": sentence,
                "score": round(val, 4),
            })
        results[fid] = examples

    return results


def compute_feature_stats(hidden: torch.Tensor) -> Dict[int, Dict]:
    """
    Compute summary statistics for every feature:
      - frequency: fraction of tokens that activate this feature at all
      - mean_activation: average activation value when active
      - max_activation: largest observed activation
      - is_dead: True if the feature never activates (frequency == 0)
    """
    n_tokens, n_features = hidden.shape
    stats = {}

    active_mask = hidden > 0
    frequencies = active_mask.float().mean(dim=0)  # (n_features,)
    max_vals = hidden.max(dim=0).values

    for fid in range(n_features):
        col = hidden[:, fid]
        active_vals = col[col > 0]
        stats[fid] = {
            "frequency": round(frequencies[fid].item(), 6),
            "mean_activation": round(active_vals.mean().item(), 4) if len(active_vals) > 0 else 0.0,
            "max_activation": round(max_vals[fid].item(), 4),
            "is_dead": bool(frequencies[fid].item() == 0),
        }
    return stats


def run_semantic_sanity_test(
    sae: SparseAutoencoder,
    model,
    tokenizer,
    layer: int,
    norm_stats_path: str,
    device: str = "cpu",
    sentence_a: str = "The attorney filed a motion in court",
    sentence_b: str = "The protein binds to the receptor",
) -> Dict:
    """
    The most important validation test (per project testing strategy):
    feed two semantically opposite sentences and check whether they
    activate disjoint top-5 feature sets.

    Returns a dict with the test result and diagnosis.
    """
    from hooks import attach_hooks, get_activation, remove_hooks

    stats = torch.load(norm_stats_path)
    attach_hooks(model, layers=[layer])

    def get_top_features(sentence: str, k: int = 5):
        tokens = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**tokens)
        act = get_activation(f"layer_{layer}")  # (1, seq_len, 768)
        act = act.mean(dim=1)  # average over tokens for a sentence-level vector
        act_norm = (act - stats["mean"]) / (stats["std"] + 1e-8)
        with torch.no_grad():
            hidden = sae.encode(act_norm.to(device))
        top = hidden[0].topk(k)
        return set(top.indices.tolist()), top.values.tolist()

    feats_a, vals_a = get_top_features(sentence_a)
    feats_b, vals_b = get_top_features(sentence_b)
    remove_hooks()

    overlap = feats_a & feats_b
    passed = len(overlap) == 0

    result = {
        "sentence_a": sentence_a,
        "sentence_b": sentence_b,
        "top5_features_a": sorted(feats_a),
        "top5_features_b": sorted(feats_b),
        "overlap": sorted(overlap),
        "passed": passed,
        "diagnosis": (
            "✅ SAE is working correctly — disjoint feature sets for unrelated concepts."
            if passed else
            "⚠️  Features overlap — lambda may be too low (not sparse enough). Consider increasing it."
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze trained SAE features")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sae_layer8.pt")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="cache")
    parser.add_argument("--sanity-test", action="store_true", help="Run legal-vs-biology sanity test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SAE checkpoint...")
    sae = load_trained_sae(args.checkpoint, device)

    logger.info("Loading cached activations and metadata...")
    acts = torch.load(f"{args.cache_dir}/layer{args.layer}_acts.pt").float()
    token_metadata = torch.load(f"{args.cache_dir}/layer{args.layer}_token_metadata.pt")
    source_texts = torch.load(f"{args.cache_dir}/layer{args.layer}_source_texts.pt")

    norm_stats_path = "checkpoints/normalization_stats.pt"

    logger.info("Computing feature activations for entire corpus...")
    hidden = compute_feature_activations(sae, acts, norm_stats_path, device=device)

    logger.info("Computing feature statistics...")
    feature_stats = compute_feature_stats(hidden)

    n_dead = sum(1 for s in feature_stats.values() if s["is_dead"])
    logger.info(f"Dead features: {n_dead}/{len(feature_stats)} ({100*n_dead/len(feature_stats):.1f}%)")

    logger.info(f"Finding top-{args.top_n} examples per feature...")
    top_examples = find_top_examples_per_feature(
        hidden, token_metadata, source_texts, top_n=args.top_n
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_dir}/feature_top_examples.json", "w") as f:
        json.dump({str(k): v for k, v in top_examples.items()}, f, indent=2)
    with open(f"{args.output_dir}/feature_stats.json", "w") as f:
        json.dump({str(k): v for k, v in feature_stats.items()}, f, indent=2)

    logger.info(f"✅ Saved: {args.output_dir}/feature_top_examples.json")
    logger.info(f"✅ Saved: {args.output_dir}/feature_stats.json")

    if args.sanity_test:
        logger.info("Running semantic sanity test (legal vs. biology)...")
        from hooks import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(device=device)
        result = run_semantic_sanity_test(
            sae, model, tokenizer, args.layer, norm_stats_path, device
        )
        print(json.dumps(result, indent=2))
        with open(f"{args.output_dir}/sanity_test_result.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
