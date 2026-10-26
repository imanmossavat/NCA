"""Leaf NCA: a neural cellular automaton that grows a veined leaf from one cell and regrows it after damage."""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac GPU: fall back to CPU for any unsupported op
