"""Provided utilities for loading the pre-tokenized TinyStories data and vocabulary.

These are given to you, not something you need to implement -- BPE tokenizer training
and encoding are not part of this assignment. You still write get_batch() yourself
(tests/test_data.py); load_memmap_dataset() below just gets you a token-ID array to
call it on.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


def load_memmap_dataset(bin_path: str | os.PathLike, meta_path: str | os.PathLike) -> np.ndarray:
    """Memory-map a pre-tokenized .bin file using the dtype/length recorded in its .meta file.

    Returns a 1-D array of token IDs suitable for passing directly to get_batch().
    """
    with open(meta_path) as f:
        meta = json.load(f)
    return np.memmap(bin_path, dtype=meta["dtype"], mode="r", shape=(meta["num_tokens"],))


def load_vocab(vocab_merges_path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load the (vocab, merges) tuple used to pre-tokenize the provided data.

    vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]] -- the same types your
    tokenizer-based classmates would have produced, had this assignment included BPE.
    """
    vocab, merges = torch.load(vocab_merges_path, weights_only=False)
    return vocab, merges


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode a sequence of token IDs back into text using the provided vocabulary.

    Concatenates each ID's byte-string vocabulary entry and decodes as UTF-8, replacing
    any malformed bytes (e.g. from an untrained/undertrained model's output) with U+FFFD.
    """
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")
