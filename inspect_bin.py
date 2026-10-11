"""
Quick inspector for nanoGPT-style .bin token files.
Helps figure out what dtype/encoding was used to create a train.bin / val.bin.

Usage:
    python inspect_bin.py path\to\train.bin
"""

import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "train.bin"

for dtype in [np.uint16, np.uint32, np.int64]:
    try:
        arr = np.fromfile(path, dtype=dtype)
        print(f"--- dtype={dtype.__name__} ---")
        print(f"  num tokens: {len(arr):,}")
        print(f"  min value: {arr.min()}")
        print(f"  max value: {arr.max()}")
        print()
    except Exception as e:
        print(f"dtype={dtype.__name__} failed: {e}")

print("Interpretation guide:")
print("  - If max value is ~50256 (and uint16 reads cleanly) -> GPT-2 BPE encoding (tiktoken).")
print("    => No meta.pkl needed. Just train directly.")
print("  - If max value is small (e.g. under a few hundred) -> likely character-level encoding.")
print("    => You need a meta.pkl with matching stoi/itos.")
