# crystal_lattice/run_crystallization.py
import os
from crystal_lattice.lattice_compiler import load_word_list, compile_lattice
from crystal_lattice.export import export_safetensors, export_jsonl

if __name__ == "__main__":
    out_dir = "out"
    limit = 150_000  # full dictionary scale

    print("[init] loading word list...")
    word_list = load_word_list(limit=limit)

    print("[compile] building lattice...")
    crystals, metadata = compile_lattice(word_list, out_dir=out_dir)

    print("[export] writing outputs...")
    export_safetensors(crystals, os.path.join(out_dir, "crystals.safetensors"))
    export_jsonl(metadata, os.path.join(out_dir, "metadata.jsonl"))

    print("[done] full crystal lattice prepared.")
