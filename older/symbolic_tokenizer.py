# crystal_lattice/symbolic_tokenizer.py
import json
from typing import Dict, List, Optional, Iterable
from .specials import SPECIAL_TOKEN_ORDER, RESERVED_BAND_SIZE

class SymbolicTokenizer:
    """
    Deterministic tokenizer with fixed special-token band [0..RESERVED_BAND_SIZE-1].
    Dictionary words begin at id_start = RESERVED_BAND_SIZE.
    """

    def __init__(self,
                 specials: Optional[List[str]] = None,
                 reserved_band_size: int = RESERVED_BAND_SIZE):
        self.reserved_band_size = int(reserved_band_size)
        self.specials = list(SPECIAL_TOKEN_ORDER if specials is None else specials)

        # Build vocab dicts
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Install specials
        for idx, tok in enumerate(self.specials):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        # Pointer for dictionary word ids
        self._next_id = max(self.token_to_id.values(), default=-1) + 1
        if self._next_id < self.reserved_band_size:
            self._next_id = self.reserved_band_size

    # -------------------------
    # basic API
    # -------------------------
    def add_tokens(self, tokens: Iterable[str]) -> None:
        for t in tokens:
            if t in self.token_to_id:
                continue
            tid = self._next_id
            self.token_to_id[t] = tid
            self.id_to_token[tid] = t
            self._next_id += 1

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        tid = self._next_id
        self.token_to_id[token] = tid
        self.id_to_token[tid] = token
        self._next_id += 1
        return tid

    def encode(self, tokens: Iterable[str]) -> List[int]:
        return [self.token_to_id.get(t, self.token_to_id.get("<unk>", 1)) for t in tokens]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    @property
    def vocab_size(self) -> int:
        return self._next_id

    @property
    def special_ids(self) -> Dict[str, int]:
        return {t: self.token_to_id[t] for t in self.specials}

    def is_special(self, token_or_id) -> bool:
        if isinstance(token_or_id, int):
            return int(token_or_id) < self.reserved_band_size
        tid = self.token_to_id.get(token_or_id, -1)
        return 0 <= tid < self.reserved_band_size

    # -------------------------
    # IO
    # -------------------------
    def to_json(self, path: str) -> None:
        data = {
            "reserved_band_size": self.reserved_band_size,
            "specials": self.specials,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path: str) -> "SymbolicTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = SymbolicTokenizer(
            specials=data.get("specials", SPECIAL_TOKEN_ORDER),
            reserved_band_size=data.get("reserved_band_size", RESERVED_BAND_SIZE),
        )
        # Restore vocab
        tok.token_to_id = dict(data["token_to_id"])
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok._next_id = max(tok.id_to_token.keys(), default=-1) + 1
        return tok
