import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import hashlib
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def _stable_seed(text):
    '''
    Deterministic 32-bit seed derived from the example text. Uses hashlib (not the builtin
    hash(), which is salted per-process via PYTHONHASHSEED) so the same review maps to the
    same transformed review across runs and machines -> reproducible evaluation.
    '''
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


# QWERTY keyboard-adjacency map: for each letter, the physically neighbouring keys.
# Used to generate *realistic* typos (the kind a human makes) rather than random noise.
_QWERTY_NEIGHBORS = {
    'q': 'wa',   'w': 'qeas',  'e': 'wrsd',  'r': 'etdf',  't': 'ryfg',
    'y': 'tugh',  'u': 'yihj',  'i': 'uojk',  'o': 'ipkl',  'p': 'ol',
    'a': 'qwsz',  's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
    'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',  'l': 'kop', 'z': 'asx',
    'x': 'zsdc',  'c': 'xdfv',  'v': 'cfgb',  'b': 'vghn', 'n': 'bhjm',  'm': 'njk',
}


def _keyboard_typo_word(word, rng):
    """
    Apply at most one realistic keyboard-adjacency edit to a single word.
    Three edit types, each a plausible human slip:
        * substitution: replace a letter with a physically adjacent key
        * insertion: insert an adjacent key next to an existing letter
        * transposition: swap two neighbouring letters (classic finger-order slip)
    The edit targets an interior letter where possible so the first/last letters
    (which humans rely on most for reading) usually stay intact, keeping the word
    interpretable and the sentiment readable.
    """
    # Only perturb purely alphabetic words long enough to tolerate one edit.
    if len(word) < 4 or not word.isalpha():
        return word

    edit_type = rng.choice(["sub", "ins", "swap"])
    chars = list(word)

    if edit_type == "swap":
        # Swap two adjacent interior characters.
        i = rng.randint(1, len(chars) - 3)  # keep first + last untouched
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    # For sub/ins, pick an interior position when possible.
    i = rng.randint(1, len(chars) - 2)
    lower = chars[i].lower()
    neighbors = _QWERTY_NEIGHBORS.get(lower)
    if not neighbors:
        return word
    repl = rng.choice(list(neighbors))
    # Preserve original casing of the target letter.
    if chars[i].isupper():
        repl = repl.upper()

    if edit_type == "sub":
        chars[i] = repl
    else:  # insertion
        chars.insert(i, repl)
    return "".join(chars)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    WORD_TYPO_PROB = 0.40  # fraction of eligible words to corrupt

    text = example["text"]

    # Seed a *local* RNG deterministically from the example content. This makes the
    # transform reproducible per-example and independent of call order / global seed.
    rng = random.Random(_stable_seed(text))

    tokens = word_tokenize(text)
    new_tokens = []
    for tok in tokens:
        if tok.isalpha() and len(tok) >= 4 and rng.random() < WORD_TYPO_PROB:
            new_tokens.append(_keyboard_typo_word(tok, rng))
        else:
            new_tokens.append(tok)

    example["text"] = TreebankWordDetokenizer().detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
