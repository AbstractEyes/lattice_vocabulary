# crystal_lattice/lattice_compiler.py
import os
import json
import numpy as np
from tqdm import tqdm
from typing import List
from .crystal_builder import generate_crystal

# ======================================================
# Word List Loader
# ======================================================
def load_word_list(source: str = "wordnet", limit: int = None) -> List[str]:
    from nltk.corpus import words
    word_list = list(set(words.words()))
    if limit:
        word_list = word_list[:limit]
    return word_list

# ======================================================
# Batch Lattice Compiler
# ======================================================
def compile_lattice(words: List[str], out_dir: str = "out", batch_size: int = 1000):
    os.makedirs(out_dir, exist_ok=True)
    all_crystals = {}
    metadata = []

    for i in tqdm(range(0, len(words), batch_size)):
        batch = words[i:i+batch_size]
        for word in batch:
            result = generate_crystal(word)
            all_crystals[word] = result["crystal"]
            metadata.append({
                "word": result["word"],
                "cardinal": result["cardinal"],
                "volume": result["volume"]
            })

    np.save(os.path.join(out_dir, "crystals.npy"), all_crystals)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return all_crystals, metadata

# ======================================================
# Entry
# ======================================================
if __name__ == "__main__":
    word_list = load_word_list(limit=100_000)
    compile_lattice(word_list)