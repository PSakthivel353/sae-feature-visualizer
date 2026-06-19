
"""
train_sae.py — Training Loop for the Sparse Autoencoder
─────────────────────────────────────────────────────────────────────────────
WHAT THIS DOES:
  1. Loads cached GPT-2 activations
  2. Normalizes them (CRITICAL — see README § Known Issues)
  3. Trains the SAE with: loss = reconstruction_MSE (sparsity enforced via Top-K)
  4. Tracks active-feature-count per epoch (should stay at target topk)
  5. Saves checkpoints + training curves + normalization stats

SPARSITY APPROACH:
  Instead of soft L1 penalty (which doesn't enforce hard sparsity limits),
  we use TOP-K: exactly K features are kept active per token, others zeroed.
  This directly controls the active feature count without tuning hyperparameters.

USAGE:
  python src/train_sae.py --epochs 10 --topk 32 --batch-size 128
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import argparse
import logging
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from sae_model import SparseAutoencoder, compute_loss
from collect_activations import load_activation_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def normalize_activations(acts: torch.Tensor) -> tuple:
    """
    Normalize activations to zero mean, unit variance per-dimension.

    CRITICAL: Without this step the SAE will fail to converge — GPT-2
    activations can have wildly different scales across dimensions, and
    an un-normalized L1 penalty will unfairly punish high-variance dims.

    Returns:
        normalized_acts, mean, std  (mean/std needed to denormalize later)
    """
    mean = acts.mean(dim=0, keepdim=True)
    std = acts.std(dim=0, keepdim=True)
    normalized = (acts - mean) / (std + 1e-8)
    return normalized, mean, std


def train(
    cache_path: str = "cache/layer8_acts.pt",
    checkpoint_dir: str = "checkpoints",
    d_hidden: int = 1024,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2e-4,
    topk: int = 32,
    val_split: float = 0.1,
    device: str = None,
    log_every: int = 50,
    patience: int = 3,
):
    """
    Main training loop.

    Args:
        cache_path:     Path to cached activation tensor (N, 768)
        checkpoint_dir: Where to save model weights
        d_hidden:       SAE hidden dimension (number of features)
        epochs:         Number of training epochs
        batch_size:     Training batch size
        lr:             Adam learning rate
        topk:           Number of top features to keep active (e.g., 32 means exactly 32 features active per token)
        val_split:      Fraction of data held out for validation
        device:         'cuda' / 'cpu' / None (auto-detect)
        log_every:      Print progress every N batches
        patience:       Stop early if validation loss stops improving
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Load and normalize data ───────────────────────────────────────────────
    logger.info(f"Loading activations from {cache_path}...")
    acts, _, _ = load_activation_cache(cache_path, device="cpu")
    acts = acts.float()
    logger.info(f"Loaded activations: shape={acts.shape}, dtype={acts.dtype}")

    acts_norm, act_mean, act_std = normalize_activations(acts)

    # Save normalization stats — required at inference time too!
    norm_stats_path = Path(checkpoint_dir) / "normalization_stats.pt"
    torch.save({"mean": act_mean, "std": act_std}, norm_stats_path)
    logger.info(f"Normalization stats saved to {norm_stats_path}")

    # ── Train / val split ─────────────────────────────────────────────────────
    n_val = int(len(acts_norm) * val_split)
    n_train = len(acts_norm) - n_val
    train_acts, val_acts = torch.utils.data.random_split(
        TensorDataset(acts_norm), [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_acts, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_acts, batch_size=batch_size, shuffle=False)

    logger.info(f"Train samples: {n_train:,} | Val samples: {n_val:,}")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    d_model = acts.shape[1]
    sae = SparseAutoencoder(
        d_model=d_model,
        d_hidden=d_hidden,
        sparsity_type="topk",
        topk=topk,
    ).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"SAE initialized: {d_model} -> {d_hidden} -> {d_model}")
    logger.info(f"Total parameters: {sae.num_parameters():,}")
    logger.info(f"Sparsity control: Top-K with k={topk} (exactly {topk} features active per token)")

    # ── Training history ──────────────────────────────────────────────────────
    history = {
        "epoch": [], "train_loss": [], "train_recon_loss": [], "train_sparsity_loss": [],
        "val_loss": [], "val_recon_loss": [], "avg_active_features": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # ── Train phase ───────────────────────────────────────────────────────
        sae.train()
        total_loss, total_recon, total_sparsity, n_batches = 0.0, 0.0, 0.0, 0
        active_counts = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (batch,) in enumerate(pbar):
            batch = batch.to(device)

            recon, hidden = sae(batch)
            loss, recon_loss, sparsity_loss = compute_loss(
                recon, batch, hidden, sparsity_type="topk"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_sparsity += sparsity_loss.item()
            n_batches += 1

            active_per_sample = (hidden > 0).float().sum(dim=1)
            active_counts.append(active_per_sample.mean().item())

            if batch_idx % log_every == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "active": f"{active_per_sample.mean().item():.1f}",
                })

        scheduler.step()

        avg_active = sum(active_counts) / len(active_counts)

        # ── Validation phase ──────────────────────────────────────────────────
        sae.eval()
        val_loss_total, val_recon_total, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, hidden = sae(batch)
                loss, recon_loss, _ = compute_loss(
                    recon, batch, hidden, sparsity_type="topk"
                )
                val_loss_total += loss.item()
                val_recon_total += recon_loss.item()
                val_batches += 1

        val_loss = val_loss_total / max(val_batches, 1)
        val_recon = val_recon_total / max(val_batches, 1)

        health_flag = "✅" if topk - 5 <= avg_active <= topk + 5 else "⚠️"

        logger.info(
            f"Epoch {epoch+1}/{epochs} │ "
            f"Train Loss: {total_loss/n_batches:.4f} │ "
            f"Val Loss: {val_loss:.4f} │ "
            f"Active Features: {avg_active:.1f} {health_flag} "
            f"(target: ~{topk})"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / n_batches)
        history["train_recon_loss"].append(total_recon / n_batches)
        history["train_sparsity_loss"].append(total_sparsity / n_batches)
        history["val_loss"].append(val_loss)
        history["val_recon_loss"].append(val_recon)
        history["avg_active_features"].append(avg_active)

        # ── Checkpoint best model ────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_path = Path(checkpoint_dir) / f"sae_layer8_best.pt"
            torch.save({
                "model_state_dict": sae.state_dict(),
                "d_model": d_model,
                "d_hidden": d_hidden,
                "epoch": epoch,
                "val_loss": val_loss,
                "topk": topk,
                "avg_active_features": avg_active,
            }, best_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping triggered after {epoch+1} epochs "
                f"(no improvement for {patience} epochs)."
            )
            break

    # ── Save final checkpoint ─────────────────────────────────────────────────
    final_path = Path(checkpoint_dir) / "sae_layer8.pt"
    torch.save({
        "model_state_dict": sae.state_dict(),
        "d_model": d_model,
        "d_hidden": d_hidden,
        "epoch": epoch,
        "val_loss": val_loss,
        "topk": topk,
        "avg_active_features": avg_active,
    }, final_path)

    history_path = Path(checkpoint_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    final_active = history["avg_active_features"][-1]
    diagnosis = (
        f"✅ Sparsity healthy." if abs(final_active - topk) < 2
        else f"⚠️  Active features {final_active:.1f} vs target {topk}."
    )

    logger.info(
        f"\n{'─'*60}\n"
        f"✅ TRAINING COMPLETE\n"
        f"   Final checkpoint: {final_path}\n"
        f"   Best checkpoint:  {checkpoint_dir}/sae_layer8_best.pt\n"
        f"   Final val loss:   {val_loss:.4f}\n"
        f"   Avg active feats: {final_active:.1f} / {d_hidden}\n"
        f"   Target (Top-K):   {topk}\n"
        f"   Diagnosis: {diagnosis}\n"
        f"{'─'*60}"
    )

    return sae, history


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder on GPT-2 activations")
    parser.add_argument("--cache-path", type=str, default="cache/layer8_acts.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--d-hidden", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--topk", type=int, default=32, help="Number of top features to keep active (directly controls sparsity)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3, help="Stop early if validation loss stops improving")
    args = parser.parse_args()

    train(
        cache_path=args.cache_path,
        checkpoint_dir=args.checkpoint_dir,
        d_hidden=args.d_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        topk=args.topk,
        val_split=args.val_split,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
