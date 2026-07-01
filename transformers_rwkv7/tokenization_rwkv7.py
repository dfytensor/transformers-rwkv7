"""RWKV-7 tokenizer — byte-level, matching the official ``rwkv_vocab_v20230424``.

Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py (RWKV_TOKENIZER)
"""

import os
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import AddedToken

VOCAB_FILES_NAMES = {"vocab_file": "rwkv_vocab_v20230424.txt"}


class Rwkv7Tokenizer(PreTrainedTokenizer):
    """Byte-level RWKV tokenizer (same vocab as RWKV-v4/v5/v6/v7 World models)."""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"RWKV vocab file not found: {vocab_file}")

        self.idx2token = {}
        sorted_tokens = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[: line.index(" ")])
                x = eval(line[line.index(" "): line.rindex(" ")])
                x = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(x, bytes)
                assert len(x) == int(line[line.rindex(" "):])
                sorted_tokens.append(x)
                self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}

        # register a pad token at id 0 (reserved — the vocab starts at 1; the embedding
        # matrix always has a row 0, so this is a safe, real vocab member)
        pad_bytes = b"<pad>"
        if 0 not in self.idx2token:
            self.idx2token[0] = pad_bytes
            self.token2idx[pad_bytes] = 0
        self._pad_id = 0

        # precompute tables for fast greedy matching
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]
        for s in reversed(sorted_tokens):  # match longer tokens first
            if len(s) >= 2:
                s0, s1 = int(s[0]), int(s[1])
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

        super().__init__(**kwargs)

    # ----- core byte-level (de)tokeniser -----
    def _encode_bytes(self, src: bytes) -> List[int]:
        tokens = []
        i, n = 0, len(src)
        while i < n:
            s = src[i: i + 1]
            if i < n - 1:
                s0, s1 = int(src[i]), int(src[i + 1])
                if s1 in self.good[s0]:
                    window = src[i: i + self.wlen[s0]]
                    try:
                        s = next(filter(window.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def _tokenize(self, text: str) -> List[str]:
        # return token ids as strings for HF compatibility
        return [str(t) for t in self._encode_bytes(text.encode("utf-8"))]

    def _convert_token_to_id(self, token: str) -> int:
        if token.isdigit():
            return int(token)
        b = token.encode("utf-8")
        if b in self.token2idx:
            return self.token2idx[b]
        return self._pad_id  # fall back to pad/unk (id 0)

    def _convert_id_to_token(self, index: int) -> str:
        b = self.idx2token.get(int(index))
        return b.decode("utf-8", errors="replace") if b is not None else "<pad>"

    def encode(self, text: str, **kwargs) -> List[int]:
        return self._encode_bytes(text.encode("utf-8"))

    def decode(self, token_ids, **kwargs) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return b"".join(self.idx2int_get(i) for i in token_ids).decode("utf-8", errors="replace")

    def idx2int_get(self, i: int) -> bytes:
        return self.idx2token.get(int(i), b"")

    @property
    def vocab_size(self) -> int:
        return len(self.idx2token)

    def get_vocab(self):
        return {t.decode("utf-8", errors="replace"): i for i, t in self.idx2token.items()}

    def save_vocabulary(self, save_directory, filename_prefix=None):
        import shutil
        os.makedirs(save_directory, exist_ok=True)
        src = None
        for k, v in getattr(self, "vocab_files_paths", {}).values() if False else []:
            pass
        # the vocab_file path is stored during init; re-derive from attrs
        out = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"])
        # we no longer keep self.vocab_file; the caller passes the same path. Try to copy if available.
        vf = getattr(self, "_vocab_file_path", None)
        if vf and os.path.isfile(vf):
            shutil.copyfile(vf, out)
            return (out,)
        return (out,)
