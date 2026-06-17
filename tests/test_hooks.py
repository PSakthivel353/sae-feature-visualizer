"""
test_hooks.py — Unit tests for GPT-2 forward hook mechanism
─────────────────────────────────────────────────────────────────────────────
NOTE: These tests download GPT-2 (~500MB) on first run via Hugging Face Hub.
Mark as slow/integration if running in CI without network access.

RUN:  pytest tests/test_hooks.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hooks import load_model_and_tokenizer, attach_hooks, get_activation, remove_hooks


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer(device="cpu")
    yield model, tokenizer
    remove_hooks()


class TestHooks:

    def test_hook_captures_correct_shape(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attach_hooks(model, layers=[8])

        tokens = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            model(**tokens)

        act = get_activation("layer_8")
        assert act is not None
        assert act.shape[0] == 1  # batch size
        assert act.shape[2] == 768  # d_model

    def test_multiple_layers_hooked(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attach_hooks(model, layers=[2, 8])

        tokens = tokenizer("Test sentence", return_tensors="pt")
        with torch.no_grad():
            model(**tokens)

        assert get_activation("layer_2") is not None
        assert get_activation("layer_8") is not None

    def test_invalid_layer_raises(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        with pytest.raises(ValueError):
            attach_hooks(model, layers=[99])

    def test_remove_hooks_clears_activations(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attach_hooks(model, layers=[8])
        tokens = tokenizer("Test", return_tensors="pt")
        with torch.no_grad():
            model(**tokens)
        remove_hooks()
        assert get_activation("layer_8") is None

    def test_tokenizer_has_pad_token(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token == tokenizer.eos_token


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
