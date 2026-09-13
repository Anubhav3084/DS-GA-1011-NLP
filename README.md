# nyu-dsga1011-a1

DS-GA 1011 Assignment 1: Linear and Neural Sentiment Classification

See the assignment handout (PDF) provided by the course for full details. This README covers
setup and the mechanics of running the code.

## Setup

Python 3.5+ and a recent version of PyTorch are required. Follow the install instructions at
https://pytorch.org/get-started/locally/ — CPU only is fine, no need for CUDA.

```sh
pip install numpy torch
```

## Getting started

To confirm everything is working, run:

```sh
python sentiment_classifier.py --model TRIVIAL
```

This should report dev accuracy `Accuracy: 444 / 872 = 0.509174`.

## Framework code

- `sentiment_classifier.py` — main entry point; reads args, loads data, trains, evaluates.
- `sentiment_data.py` — data reading, `SentimentExample`, `WordEmbeddings`.
- `utils.py` — `Indexer`, a bijective mapping between indices and string features.
- `ffnn_example.py` — minimal PyTorch example (XOR) showing the network/train/eval pattern.
- `models.py` — **the file you modify.** Implement `train_linear_model` (Part 1: logistic
  regression) and `train_deep_averaging_network` (Part 2: deep averaging network).
- `data/` — `train.txt`, `dev.txt`, `test-blind.txt`, and two GloVe embedding files
  (`glove.6B.50d-relativized.txt`, `glove.6B.300d-relativized.txt`).

## Before you submit

You will submit only `models.py` to Gradescope. Make sure both of these run successfully:

```sh
python sentiment_classifier.py --model LR --feats UNIGRAM
python sentiment_classifier.py --model DAN
```
