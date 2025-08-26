# crystal_lattice/lattice_compiler.py (drop-in replacement)
import os, json, numpy as np
from tqdm import tqdm
from typing import List, Dict
from .crystal_builder import generate_crystal
from .symbolic_tokenizer import SymbolicTokenizer
from .specials import SPECIAL_TOKEN_ORDER, RESERVED_BAND_SIZE

def compile_lattice(words: List[str],
                    out_dir: str = "out",
                    batch_size: int = 1000,
                    save_tokenizer: bool = True) -> Dict[str, np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build tokenizer
    tok = SymbolicTokenizer()
    # Reserve first 1000 IDs for specials (already installed)
    # Add dictionary words (start at id >= 1000)
    tok.add_tokens(words)

    # 2) Generate crystals (specials + words)
    crystals = {}
    metadata = []
    # Specials first, deterministic order
    all_tokens = list(SPECIAL_TOKEN_ORDER) + words

    for i in tqdm(range(0, len(all_tokens), batch_size), desc="crystallizing"):
        batch = all_tokens[i:i+batch_size]
        for token in batch:
            try:
                res = generate_crystal(token)
                crystals[token] = res["crystal"]
                meta = {
                    "token": token,
                    "token_id": tok.token_to_id[token],
                    "is_special": res.get("is_special", False),
                    "cardinal": res.get("cardinal"),
                    "volume": float(res.get("volume")),
                }
                metadata.append(meta)
            except Exception as e:
                print(f"[skip] {token} → {e}")

    # 3) Save outputs
    np.save(os.path.join(out_dir, "crystals.npy"), crystals)
    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    if save_tokenizer:
        tok.to_json(os.path.join(out_dir, "symbolic_tokenizer.json"))

    print(f"[done] vocab={tok.vocab_size} | specials={len(SPECIAL_TOKEN_ORDER)} | words={len(words)}")
    return crystals

if __name__ == "__main__":
    # Example: plug your loader here
    try:
        from .word_loader import load_wordlist
        words = load_wordlist(source="words", limit=100_000)  # or "wordnet" / "local_json"
    except Exception:
        words = []  # empty fallback
    compile_lattice(words)
