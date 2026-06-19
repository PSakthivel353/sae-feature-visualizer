"""
sae_model.py — Sparse Autoencoder Architecture
─────────────────────────────────────────────────────────────────────────────
ARCHITECTURE:

    Input (768)  ──Linear+ReLU──>  Hidden (4096, SPARSE)  ──Linear──>  Recon (768)
    [GPT-2 activation]              [interpretable features]          [reconstruction]

  The hidden layer is wider than the input (4096 > 768) — this is called an
  "overcomplete" or "expansion" dictionary. Counterintuitively, making the
  hidden layer LARGER while forcing it to be SPARSE is what makes individual
  features become interpretable. Each of the 4096 hidden units becomes a
  candidate "feature direction" that may correspond to a human concept.

  Without sparsity, a small hidden layer would just learn a compressed,
  entangled (superposed) representation — exactly the opaque thing we are
  trying to undo.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SparseAutoencoder(nn.Module):
    """
    A Sparse Autoencoder (SAE) for decomposing transformer activations
    into interpretable, (mostly) monosemantic features.

    Args:
        d_model:  Dimensionality of input activations (768 for GPT-2 base)
        d_hidden: Dimensionality of the sparse hidden layer (4096 = 5.33x expansion)
        tied_weights: If True, decoder weights = encoder weights transposed
                       (reduces parameters, sometimes improves training stability)
        sparsity_type: 'l1' (soft penalty) or 'topk' (hard cutoff)
        topk: Number of top features to keep active (only used if sparsity_type='topk')
    """

    def __init__(
        self,
        d_model: int = 768,
        d_hidden: int = 4096,
        tied_weights: bool = False,
        sparsity_type: str = "l1",
        topk: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.tied_weights = tied_weights
        self.sparsity_type = sparsity_type
        self.topk = topk

        self.encoder = nn.Linear(d_model, d_hidden, bias=True)

        if sparsity_type == "l1":
            self.relu = nn.ReLU()
        elif sparsity_type != "topk":
            raise ValueError(f"Unknown sparsity_type: {sparsity_type}")

        if tied_weights:
            # Decoder reuses encoder weights (transposed) — saves params
            self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.decoder = nn.Linear(d_hidden, d_model, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Standard SAE initialization: decoder columns unit-norm."""
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder.bias)

        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
            with torch.no_grad():
                self.decoder.weight.data = (
                    self.decoder.weight.data
                    / self.decoder.weight.data.norm(dim=0, keepdim=True)
                )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map activation vector -> sparse feature vector.

        Args:
            x: (batch, d_model)
        Returns:
            hidden: (batch, d_hidden), sparse with exactly topk nonzeros (if topk mode)
        """
        pre_activation = self.encoder(x)

        if self.sparsity_type == "l1":
            return torch.relu(pre_activation)
        elif self.sparsity_type == "topk":
            # Keep only the top-k activations, zero out the rest
            topk_vals, topk_idxs = torch.topk(pre_activation, self.topk, dim=1, largest=True)
            hidden = torch.zeros_like(pre_activation)
            hidden.scatter_(1, topk_idxs, topk_vals)
            return hidden
        else:
            raise ValueError(f"Unknown sparsity_type: {self.sparsity_type}")

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Map sparse feature vector -> reconstructed activation.

        Args:
            hidden: (batch, d_hidden)
        Returns:
            recon: (batch, d_model)
        """
        if self.tied_weights:
            return hidden @ self.encoder.weight + self.decoder_bias
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: (batch, d_model) — raw or normalized GPT-2 activations
        Returns:
            recon:  (batch, d_model) — reconstructed activation
            hidden: (batch, d_hidden) — sparse feature activations
        """
        hidden = self.encode(x)
        recon = self.decode(hidden)
        return recon, hidden

    @torch.no_grad()
    def get_feature_direction(self, feature_id: int) -> torch.Tensor:
        """
        Get the decoder direction (output vector) for a given feature.
        This is the direction in activation-space that this feature "writes to"
        when active. Useful for understanding what a feature represents.
        """
        if self.tied_weights:
            return self.encoder.weight[feature_id]
        return self.decoder.weight[:, feature_id]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    hidden: torch.Tensor,
    l1_lambda: float = 1e-3,
    sparsity_type: str = "l1",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the SAE training loss:
      - If sparsity_type='l1': L = MSE(recon, target) + lambda * mean(|hidden|)
      - If sparsity_type='topk': L = MSE(recon, target)  (sparsity is enforced directly)

    Args:
        recon:          (batch, d_model) reconstructed activations
        target:         (batch, d_model) original (normalized) activations
        hidden:         (batch, d_hidden) sparse feature activations
        l1_lambda:      sparsity penalty weight (only used for L1 mode)
        sparsity_type:  'l1' or 'topk'

    Returns:
        total_loss, recon_loss, sparsity_loss  (all scalars)
    """
    recon_loss = ((recon - target) ** 2).mean()

    if sparsity_type == "topk":
        # Top-K mode: sparsity is direct, no soft penalty needed
        sparsity_loss = torch.tensor(0.0, device=recon.device)
        total_loss = recon_loss
    else:
        # L1 mode: soft sparsity penalty
        sparsity_loss = hidden.abs().mean()
        total_loss = recon_loss + l1_lambda * sparsity_loss

    return total_loss, recon_loss, sparsity_loss


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sae = SparseAutoencoder(d_model=768, d_hidden=4096)
    print(f"SAE parameters: {sae.num_parameters():,}")

    dummy_input = torch.randn(32, 768)  # batch of 32 fake activations
    recon, hidden = sae(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Recon shape:  {recon.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"Active features per sample (avg): {(hidden > 0).float().sum(dim=1).mean():.1f} / {sae.d_hidden}")

    loss, recon_loss, sparsity_loss = compute_loss(recon, dummy_input, hidden)
    print(f"Total loss: {loss.item():.4f} (recon: {recon_loss.item():.4f}, sparsity: {sparsity_loss.item():.4f})")
