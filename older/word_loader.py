import json
from typing import List

def load_wordlist(source: str = "wordnet", limit: int = None) -> List[str]:
    """
    Loads a list of words from a specified source.

    Args:
        source (str): "wordnet", "local", or "huggingface"
        limit (int): optional maximum number of words to return

    Returns:
        List[str]: list of valid words
    """
    if source == "wordnet":
        from nltk.corpus import wordnet
        word_set = set()
        for synset in wordnet.all_synsets():
            word_set.update(lemma.name().lower() for lemma in synset.lemmas())
        words = sorted(word_set)

    elif source == "words":  # Simple nltk words corpus
        from nltk.corpus import words as nltk_words
        words = sorted(set(nltk_words.words()))

    elif source == "local_json":
        with open("dictionary_words.json", "r") as f:
            words = json.load(f)  # Must be a list of strings

    elif source == "hf":
        raise NotImplementedError("Hugging Face loader requires `datasets` integration")

    else:
        raise ValueError(f"Unknown source: {source}")

    if limit:
        return words[:limit]
    return words
