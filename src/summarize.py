import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import sys
import types
from importlib.machinery import ModuleSpec
# Stub optional backends to prevent heavy imports in mixed envs
for _mod in ("tensorflow", "jax", "flax", "torchvision"):
    if _mod not in sys.modules:
        m = types.ModuleType(_mod)
        m.__spec__ = ModuleSpec(_mod, loader=None)
        m.__path__ = []  # mark as package-like
        sys.modules[_mod] = m
        # Special handling for torchvision.transforms
        if _mod == "torchvision":
            tmod = types.ModuleType("torchvision.transforms")
            tmod.__spec__ = ModuleSpec("torchvision.transforms", loader=None)
            class _InterpolationMode:
                NEAREST = 0
                BILINEAR = 1
                BICUBIC = 2
                BOX = 3
                HAMMING = 4
                LANCZOS = 5
            tmod.InterpolationMode = _InterpolationMode
            sys.modules["torchvision.transforms"] = tmod
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "pszemraj/long-t5-tglobal-base-16384-book-summary"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


def summarize(text: str, max_length: int = 256, min_length: int = 80, num_beams: int = 4) -> str:
    """
    Abstractive summarization using Long-T5 (CPU-friendly settings).
    """
    # Add task prefix for T5-style models
    prompt = f"summarize: {text}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
