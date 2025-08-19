# crystal_lattice/export.py
import os
import json
import numpy as np
from safetensors.numpy import save_file

# ======================================================
# Safetensors Export
# ======================================================
def export_safetensors(crystals: dict, out_path: str):
    tensor_map = {}
    for word, crystal in crystals.items():
        key = f"crystal::{word}"
        tensor_map[key] = crystal.astype(np.float32)
    save_file(tensor_map, out_path)
    print(f"[export] wrote {len(tensor_map)} entries → {out_path}")

# ======================================================
# JSONL Export (metadata only)
# ======================================================
def export_jsonl(metadata: list, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    print(f"[export] wrote metadata → {out_path}")

# ======================================================
# Entry
# ======================================================
if __name__ == "__main__":
    crystals = np.load("out/crystals.npy", allow_pickle=True).item()
    with open("out/metadata.json", "r") as f:
        metadata = json.load(f)

    export_safetensors(crystals, "out/crystals.safetensors")
    export_jsonl(metadata, "out/metadata.jsonl")
