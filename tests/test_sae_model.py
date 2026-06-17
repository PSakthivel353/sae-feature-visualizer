"""
test_sae_model.py — Unit tests for the SAE architecture
─────────────────────────────────────────────────────────────────────────────
RUN:  pytest tests/ -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sae_model import SparseAutoencoder, compute_loss


class TestSparseAutoencoder:

    def test_output_shapes(self):
        sae = SparseAutoencoder(d_model=768, d_hidden=4096)
        x = torch.randn(16, 768)
        recon, hidden = sae(x)
        assert recon.shape == (16, 768)
        assert hidden.shape == (16, 4096)

    def test_hidden_is_nonnegative(self):
        """ReLU guarantees sparsity is achievable — all hidden values >= 0."""
        sae = SparseAutoencoder(d_model=768, d_hidden=4096)
        x = torch.randn(32, 768) * 5  # Large variance input
        _, hidden = sae(x)
        assert (hidden >= 0).all()

    def test_encode_decode_consistency(self):
        sae = SparseAutoencoder(d_model=768, d_hidden=4096)
        x = torch.randn(8, 768)
        hidden = sae.encode(x)
        recon = sae.decode(hidden)
        recon_direct, hidden_direct = sae(x)
        assert torch.allclose(recon, recon_direct, atol=1e-6)
        assert torch.allclose(hidden, hidden_direct, atol=1e-6)

    def test_different_dimensions(self):
        sae = SparseAutoencoder(d_model=128, d_hidden=512)
        x = torch.randn(4, 128)
        recon, hidden = sae(x)
        assert recon.shape == (4, 128)
        assert hidden.shape == (4, 512)

    def test_tied_weights(self):
        sae = SparseAutoencoder(d_model=768, d_hidden=4096, tied_weights=True)
        x = torch.randn(4, 768)
        recon, hidden = sae(x)
        assert recon.shape == (4, 768)
        assert not hasattr(sae, "decoder")

    def test_get_feature_direction(self):
        sae = SparseAutoencoder(d_model=768, d_hidden=4096)
        direction = sae.get_feature_direction(0)
        assert direction.shape == (768,)

    def test_num_parameters_positive(self):
        sae = SparseAutoencoder(d_model=768, d_hidden=4096)
        assert sae.num_parameters() > 0
        # encoder: 768*4096 + 4096, decoder: 4096*768 + 768
        expected = (768 * 4096 + 4096) + (4096 * 768 + 768)
        assert sae.num_parameters() == expected


class TestLossFunction:

    def test_loss_is_scalar(self):
        recon = torch.randn(16, 768)
        target = torch.randn(16, 768)
        hidden = torch.relu(torch.randn(16, 4096))
        loss, recon_loss, sparsity_loss = compute_loss(recon, target, hidden, l1_lambda=1e-3)
        assert loss.dim() == 0
        assert recon_loss.dim() == 0
        assert sparsity_loss.dim() == 0

    def test_perfect_reconstruction_zero_recon_loss(self):
        x = torch.randn(16, 768)
        hidden = torch.zeros(16, 4096)
        loss, recon_loss, sparsity_loss = compute_loss(x, x, hidden, l1_lambda=1e-3)
        assert recon_loss.item() < 1e-6
        assert sparsity_loss.item() == 0.0

    def test_higher_lambda_increases_loss_contribution(self):
        recon = torch.randn(16, 768)
        target = torch.randn(16, 768)
        hidden = torch.relu(torch.randn(16, 4096)) + 1.0  # force nonzero activations
        loss_low, _, _ = compute_loss(recon, target, hidden, l1_lambda=1e-4)
        loss_high, _, _ = compute_loss(recon, target, hidden, l1_lambda=1e-1)
        assert loss_high.item() > loss_low.item()

    def test_sparsity_loss_is_mean_abs_hidden(self):
        hidden = torch.tensor([[1.0, 0.0, 2.0, 0.0]])
        recon = torch.zeros(1, 768)
        target = torch.zeros(1, 768)
        _, _, sparsity_loss = compute_loss(recon, target, hidden, l1_lambda=1.0)
        assert abs(sparsity_loss.item() - 0.75) < 1e-6  # mean(|1,0,2,0|) = 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
