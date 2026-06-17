"""
download_corpus.py — Fetch and prepare a text corpus for activation collection
─────────────────────────────────────────────────────────────────────────────
Downloads a sample from "The Pile" (uncopyrighted variant) or OpenWebText
via Hugging Face `datasets`, and writes it to data/pile_sample.jsonl in the
exact format collect_activations.py expects: one JSON object per line with
a "text" field.

USAGE:
  python scripts/download_corpus.py --source pile --n-samples 50000
  python scripts/download_corpus.py --source openwebtext --n-samples 50000
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SOURCES = {
    "pile": "monology/pile-uncopyrighted",
    "openwebtext": "Skylion007/openwebtext",
}


def download_corpus(source: str, n_samples: int, output_path: str, min_length: int = 50):
    from datasets import load_dataset

    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Choose from: {list(SOURCES.keys())}")

    dataset_name = SOURCES[source]
    logger.info(f"Streaming dataset: {dataset_name}")

    ds = load_dataset(dataset_name, split="train", streaming=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=n_samples, desc="Downloading corpus")
        for example in ds:
            text = example.get("text", "").strip()
            if len(text) < min_length:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            written += 1
            pbar.update(1)
            if written >= n_samples:
                break
        pbar.close()

    logger.info(f"✅ Saved {written:,} documents to {output_path}")
    size_mb = Path(output_path).stat().st_size / 1e6
    logger.info(f"   File size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download a text corpus for SAE training")
    parser.add_argument("--source", type=str, default="pile", choices=list(SOURCES.keys()))
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--output", type=str, default="data/pile_sample.jsonl")
    parser.add_argument("--min-length", type=int, default=50)
    args = parser.parse_args()

    download_corpus(args.source, args.n_samples, args.output, args.min_length)


if __name__ == "__main__":
    main()
