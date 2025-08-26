# crystal_lattice/cardinal_design.py
import re

# ======================================================
# Canonical ℵ Classifier (Transfinite Strata)
# ======================================================
def assign_cardinal(word: str) -> str:
    # ℵ₀ → base symbolic identity (simple nouns, verbs)
    if re.fullmatch(r"[a-z]+", word):
        return "ℵ₀"

    # ℵ₁ → morphologically derived (e.g., running, re-*, -ing, -ness)
    if any(suffix in word for suffix in ["ing", "ed", "ness", "ment", "tion"]):
        return "ℵ₁"
    if any(prefix in word for prefix in ["re", "un", "non", "pre", "post"]):
        return "ℵ₁"

    # ℵ₂ → meta-structure, proper nouns, acronyms
    if word.isupper():
        return "ℵ₂"
    if word[0].isupper():
        return "ℵ₂"

    # ℵ₃ → syntactic/function markers (aux verbs, prepositions, symbols)
    if word in {"the", "of", "to", "and", "in", "for", "with", "on", "at", "by", "from"}:
        return "ℵ₃"

    return "ℵ₀"

# ======================================================
# Override Hook (future injection)
# ======================================================
def assign_cardinal_custom(word: str, custom_rules: dict = None) -> str:
    if custom_rules and word in custom_rules:
        return custom_rules[word]
    return assign_cardinal(word)
