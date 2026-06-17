"""
label_features.py — Interactive CLI for Manually Labeling Features
─────────────────────────────────────────────────────────────────────────────
Per the project build timeline (Week 3: "Feature labeling"), this script
walks you through candidate features (sorted by frequency, so you see the
most "active" / informative ones first) and lets you assign a human-readable
label after reading their top-activating examples.

Labels are saved to cache/feature_labels.json:
  {"312": "legal/courtroom language", "847": "negation (not, never, no)"}

This file is then consumed by the dashboard to show readable names instead
of just numeric IDs (optional enhancement — see README § Future Extensions).

USAGE:
  python scripts/label_features.py --n-candidates 30
─────────────────────────────────────────────────────────────────────────────
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Manually label SAE features")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--n-candidates", type=int, default=30)
    parser.add_argument("--min-examples", type=int, default=5,
                         help="Skip features with fewer than this many top examples")
    args = parser.parse_args()

    examples_path = Path(args.cache_dir) / "feature_top_examples.json"
    stats_path = Path(args.cache_dir) / "feature_stats.json"
    labels_path = Path(args.cache_dir) / "feature_labels.json"

    if not examples_path.exists() or not stats_path.exists():
        print(f"❌ Missing {examples_path} or {stats_path}.")
        print("   Run feature_analysis.py first.")
        return

    with open(examples_path) as f:
        examples = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)

    labels = {}
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    # Sort candidates by frequency descending (most informative first),
    # skipping dead features and ones with too few examples.
    candidates = [
        fid for fid, s in sorted(stats.items(), key=lambda x: -x[1]["frequency"])
        if not s.get("is_dead") and len(examples.get(fid, [])) >= args.min_examples
        and fid not in labels
    ][: args.n_candidates]

    print(f"\n{'='*70}\n SAE FEATURE LABELING — {len(candidates)} candidates\n{'='*70}")
    print("For each feature, read the top examples, then type a short label.")
    print("Commands: [Enter] = skip, 'q' = quit and save\n")

    for fid in candidates:
        exs = examples[fid][:8]
        print(f"\n--- Feature {fid} (frequency: {stats[fid]['frequency']*100:.3f}%) ---")
        for e in exs:
            token = e["token"].replace("Ġ", " ").strip()
            sentence_snippet = e["sentence"][:80].replace("\n", " ")
            print(f"  [{e['score']:.3f}] '{token}'  ←  {sentence_snippet}...")

        label = input("\n  Label this feature (or Enter to skip, 'q' to quit): ").strip()
        if label.lower() == "q":
            break
        if label:
            labels[fid] = label
            print(f"  ✅ Saved: Feature {fid} = '{label}'")

    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\n✅ {len(labels)} total labels saved to {labels_path}")


if __name__ == "__main__":
    main()
