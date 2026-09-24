# nyu-dsga1011-a2

DS-GA 1011 Assignment 2: Building a Transformer LM

This assignment is adapted from Stanford CS336 Assignment 1, via NYU's "Building LLM
Reasoners" course. All credit for the original development goes to the Stanford course
staff. See the assignment handout (distributed separately) for the full assignment
description.

Note: this assignment does **not** have you build a BPE tokenizer or tokenize raw text
yourself. Instead, a small amount of pre-tokenized TinyStories data (and the vocabulary
used to produce it) is provided directly in this repo -- see "Provided data" below.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Provided data

`data/` already contains everything you need -- there's nothing to download:

- `tinystories_valid.bin` + `.meta`: the TinyStories validation split, pre-tokenized with
  a 10,000-token byte-level BPE vocabulary (~5.4M tokens, `uint16`).
- `tinystories_vocab_merges.pt`: the `(vocab, merges)` used to produce the above, for
  turning generated token IDs back into text.

[`data_utils.py`](./data_utils.py) has two small provided functions built on top of these:

- `load_memmap_dataset(bin_path, meta_path)` -- memory-maps a `.bin` file and returns a
  1-D array of token IDs, ready to pass to the `get_batch` function you'll implement.
- `load_vocab(vocab_merges_path)` and `decode(ids, vocab)` -- for turning your model's
  sampled token IDs back into text.

## Repo layout

```
student/*         -- this is where you write your code. Structure it however you want.
tests/adapters.py  -- glue code connecting your implementation to our tests. Fill this in.
tests/test_*.py    -- the tests you must pass. Don't edit these.
data/              -- provided pre-tokenized data + vocab (see above).
data_utils.py      -- provided helpers for loading the data above.
```
