import os
from collections import Counter

from transformers import T5TokenizerFast

from load_data import (
    load_lines, TASK_PREFIX, MAX_SOURCE_LEN, MAX_TARGET_LEN, T5_CHECKPOINT,
)

DATA_DIR = 'data'
SPLITS = ['train', 'dev']


def whitespace_stats(lines):
    """BEFORE preprocessing: whitespace-word tokenization, no truncation/prefix."""
    lengths = []
    vocab = set()
    for line in lines:
        words = line.split()
        lengths.append(len(words))
        vocab.update(words)
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    return mean_len, len(vocab)


def tokenizer_stats(lines, tokenizer, max_len, prefix=''):
    """AFTER preprocessing: T5 tokenizer, with prefix + truncation as used in training."""
    lengths = []
    vocab = set()
    for line in lines:
        ids = tokenizer(prefix + line, truncation=True, max_length=max_len,
                        return_attention_mask=False)['input_ids']
        lengths.append(len(ids))
        vocab.update(ids)
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    return mean_len, len(vocab)


def main():
    tokenizer = T5TokenizerFast.from_pretrained(T5_CHECKPOINT)

    print(f"Pretrained tokenizer: {T5_CHECKPOINT}")
    print(f"Task prefix (NL inputs): '{TASK_PREFIX}'")
    print(f"MAX_SOURCE_LEN={MAX_SOURCE_LEN}, MAX_TARGET_LEN={MAX_TARGET_LEN}\n")

    header = (
        f"{'split':<6} {'n':>6} | "
        f"{'NL words(mean)':>14} {'NL wvocab':>10} {'SQL words(mean)':>15} {'SQL wvocab':>11} | "
        f"{'NL tok(mean)':>12} {'NL tvocab':>10} {'SQL tok(mean)':>13} {'SQL tvocab':>11}"
    )
    print("=" * len(header))
    print("BEFORE (whitespace words)  |  AFTER (T5 tokenizer tokens)")
    print(header)
    print("-" * len(header))

    rows = {}
    for split in SPLITS:
        nl = load_lines(os.path.join(DATA_DIR, f'{split}.nl'))
        sql = load_lines(os.path.join(DATA_DIR, f'{split}.sql'))
        n = len(nl)

        # BEFORE: whitespace words
        nl_w_mean, nl_w_vocab = whitespace_stats(nl)
        sql_w_mean, sql_w_vocab = whitespace_stats(sql)

        # AFTER: tokenizer tokens (NL gets the task prefix; SQL does not)
        nl_t_mean, nl_t_vocab = tokenizer_stats(nl, tokenizer, MAX_SOURCE_LEN, prefix=TASK_PREFIX)
        sql_t_mean, sql_t_vocab = tokenizer_stats(sql, tokenizer, MAX_TARGET_LEN, prefix='')

        rows[split] = dict(
            n=n,
            nl_w_mean=nl_w_mean, nl_w_vocab=nl_w_vocab,
            sql_w_mean=sql_w_mean, sql_w_vocab=sql_w_vocab,
            nl_t_mean=nl_t_mean, nl_t_vocab=nl_t_vocab,
            sql_t_mean=sql_t_mean, sql_t_vocab=sql_t_vocab,
        )

        print(
            f"{split:<6} {n:>6} | "
            f"{nl_w_mean:>14.2f} {nl_w_vocab:>10} {sql_w_mean:>15.2f} {sql_w_vocab:>11} | "
            f"{nl_t_mean:>12.2f} {nl_t_vocab:>10} {sql_t_mean:>13.2f} {sql_t_vocab:>11}"
        )

    print("=" * len(header))
    print("\nLegend:")
    print("  words(mean) = mean whitespace-word count per sequence (BEFORE preprocessing)")
    print("  wvocab      = number of distinct whitespace words (BEFORE preprocessing)")
    print("  tok(mean)   = mean T5-tokenizer-token count per sequence (AFTER preprocessing)")
    print("  tvocab      = number of distinct T5-tokenizer token ids used (AFTER preprocessing)")
    return rows


if __name__ == '__main__':
    main()
