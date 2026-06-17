"""
hooks.py — Intercept GPT-2 Internal Activations via PyTorch Forward Hooks
─────────────────────────────────────────────────────────────────────────────
HOW THIS WORKS:
  GPT-2 is a transformer with 12 layers. Each layer's output is a tensor of
  shape (batch_size, sequence_length, 768). A "forward hook" is a callback
  that PyTorch fires AUTOMATICALLY every time a layer finishes computing —
  without modifying the model's source code at all.

  We attach a hook to layer 8 (index 8 of model.h). When GPT-2 processes
  a sentence, the hook fires and we silently copy the activation tensor to
  our `activations` dictionary for later use.

  Think of it like placing a transparent tap on a water pipe — you capture
  the flow without altering it.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
from transformers import GPT2Model, GPT2Tokenizer
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ─── Global activation store ─────────────────────────────────────────────────
# Keys: layer name strings (e.g. 'layer_8')
# Values: activation tensor from that layer's last forward pass
activations: Dict[str, torch.Tensor] = {}

# ─── Hook handles (needed to remove hooks later) ─────────────────────────────
_hook_handles: List = []


def make_hook(layer_name: str):
    """
    Factory function that returns a hook closure for the given layer name.

    The hook captures:
      - module: the nn.Module being hooked (GPT-2 transformer block)
      - input:  tuple of inputs to the module
      - output: tuple of outputs from the module
                output[0] is shape (batch, seq_len, d_model=768)
    """
    def hook(module, input, output):
        # output for GPT-2 transformer blocks is a tuple;
        # output[0] is the hidden state tensor (batch, seq_len, 768)
        tensor = output[0] if isinstance(output, tuple) else output
        activations[layer_name] = tensor.detach().cpu()
    return hook


def load_model_and_tokenizer(
    model_name: str = "gpt2",
    device: Optional[str] = None
) -> Tuple[GPT2Model, GPT2Tokenizer]:
    """
    Load GPT-2 model and tokenizer from Hugging Face.

    Args:
        model_name: Hugging Face model identifier.
                    Options: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        device:     'cuda', 'cpu', or None (auto-detect)

    Returns:
        (model, tokenizer) tuple ready for inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device}...")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)

    # GPT-2 tokenizer has no pad token by default — must set it
    # or batching will raise a warning / error
    tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()  # Disable dropout for deterministic activations

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def attach_hooks(model: GPT2Model, layers: Optional[List[int]] = None) -> None:
    """
    Attach forward hooks to specified transformer layers.

    Args:
        model:  GPT-2 model instance
        layers: List of layer indices to hook (0-11 for gpt2).
                Default: [8] — recommended starting layer.

    After calling this, any forward pass through `model` will populate
    the global `activations` dict with keys like 'layer_8'.
    """
    global _hook_handles
    remove_hooks()  # Clean up any previously attached hooks

    if layers is None:
        layers = [8]

    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(model.h):
            raise ValueError(
                f"Layer index {layer_idx} out of range. "
                f"GPT-2 has {len(model.h)} layers (0–{len(model.h)-1})."
            )
        handle = model.h[layer_idx].register_forward_hook(
            make_hook(f"layer_{layer_idx}")
        )
        _hook_handles.append(handle)
        logger.debug(f"Hook attached to layer {layer_idx}")

    logger.info(f"Hooks active on layers: {layers}")


def remove_hooks() -> None:
    """Remove all currently registered hooks to avoid memory leaks."""
    global _hook_handles, activations
    for handle in _hook_handles:
        handle.remove()
    _hook_handles.clear()
    activations.clear()
    logger.debug("All hooks removed and activations cleared.")


def get_activation(layer_name: str) -> Optional[torch.Tensor]:
    """
    Retrieve the activation tensor for a specific layer after a forward pass.

    Args:
        layer_name: e.g. 'layer_8'

    Returns:
        Tensor of shape (batch, seq_len, d_model) or None if not yet populated
    """
    return activations.get(layer_name, None)


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model, tokenizer = load_model_and_tokenizer()
    attach_hooks(model, layers=[8])

    test_sentence = "The lawyer argued brilliantly in front of the judge."
    tokens = tokenizer(test_sentence, return_tensors="pt")

    with torch.no_grad():
        model(**tokens)

    act = get_activation("layer_8")
    print(f"\n✅ Hook test PASSED")
    print(f"   Input sentence : '{test_sentence}'")
    print(f"   Activation shape: {act.shape}")   # Expected: (1, num_tokens, 768)
    print(f"   Value range    : [{act.min():.3f}, {act.max():.3f}]")
    print(f"   Tokens         : {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")

    remove_hooks()
    print("   Hooks cleaned up.")
