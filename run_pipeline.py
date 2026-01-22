import os
import json

# Bootstrap to avoid optional backend import issues before Transformers loads
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import sys
import types
from importlib.machinery import ModuleSpec

def _stub_optional_backends():
    # tensorflow stub with Tensor attr
    if "tensorflow" not in sys.modules:
        tf = types.ModuleType("tensorflow")
        tf.__spec__ = ModuleSpec("tensorflow", loader=None)
        tf.__path__ = []
        class Tensor:  # minimal placeholder
            pass
        tf.Tensor = Tensor
        sys.modules["tensorflow"] = tf
    # jax/flax stubs
    for _mod in ("jax", "flax"):
        if _mod not in sys.modules:
            m = types.ModuleType(_mod)
            m.__spec__ = ModuleSpec(_mod, loader=None)
            m.__path__ = []
            sys.modules[_mod] = m
    # torchvision plus transforms.InterpolationMode
    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv.__spec__ = ModuleSpec("torchvision", loader=None)
        tv.__path__ = []
        sys.modules["torchvision"] = tv
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

_stub_optional_backends()

from src.chunk import chunk_text
from src.build_rag import build_rag

# Resolve paths relative to this file so running from any CWD works
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "sample.txt")
OUT_DIR = os.path.join(BASE_DIR, "data", "rag")
OUT_PATH = os.path.join(OUT_DIR, "rag_dataset.jsonl")


def ensure_dirs():
    os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "chunks"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "rag"), exist_ok=True)


def main():
    ensure_dirs()

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    rag_data = build_rag(chunks, source="sample")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in rag_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("RAG dataset created:", len(rag_data), "chunks")
    print("Output:", OUT_PATH)


if __name__ == "__main__":
    main()
