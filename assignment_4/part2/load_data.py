import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
# punkt is part of the starter imports; download quietly and never fail hard if offline
# (this module tokenizes with the T5 tokenizer, not nltk, so it is not on the critical path).
try:
    nltk.download('punkt', quiet=True)
except Exception:
    pass
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# Pretrained checkpoint used for both the tokenizer and the model.
T5_CHECKPOINT = 'google-t5/t5-small'

# Task prefix prepended to every natural-language input. T5 is a text-to-text model and
# benefits from a short instruction-style prefix describing the task.
TASK_PREFIX = 'translate English to SQL: '

# Maximum sequence lengths (in tokenizer tokens). Chosen from the dataset statistics
# (see dataset_stats.py / report Table 2): NL inputs are short; SQL targets are long.
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 512


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Data processing for the T5 text-to-SQL model.

        * Uses the 'google-t5/t5-small' tokenizer for both encoder input (natural language)
          and decoder output (SQL).
        * The decoder is fed a start token: we use the T5 decoder_start_token_id, which is
          the pad token (id 0) for T5. This is the conventional "shift-right" setup.
        * Behavior differs on the test split: the test set has no gold SQL, so we only
          build/tokenize the encoder inputs there.
        '''
        self.split = split
        self.is_test = (split == 'test')
        self.tokenizer = T5TokenizerFast.from_pretrained(T5_CHECKPOINT)
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)
        self.nl_texts = nl_lines

        # Encoder inputs: prefix + natural language, tokenized with truncation.
        self.encoder_ids = []
        for nl in nl_lines:
            enc = tokenizer(
                TASK_PREFIX + nl,
                truncation=True,
                max_length=MAX_SOURCE_LEN,
                return_attention_mask=False,
            )['input_ids']
            self.encoder_ids.append(torch.tensor(enc, dtype=torch.long))

        if not self.is_test:
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            self.sql_texts = sql_lines
            # Decoder target ids: tokenized SQL (includes the EOS token </s>).
            self.target_ids = []
            for sql in sql_lines:
                tgt = tokenizer(
                    sql,
                    truncation=True,
                    max_length=MAX_TARGET_LEN,
                    return_attention_mask=False,
                )['input_ids']
                self.target_ids.append(torch.tensor(tgt, dtype=torch.long))
        else:
            self.sql_texts = None
            self.target_ids = None

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.is_test:
            return {'encoder_ids': self.encoder_ids[idx]}
        return {
            'encoder_ids': self.encoder_ids[idx],
            'target_ids': self.target_ids[idx],
        }


# T5's decoder start token id is the pad token id (0) for the t5-small config.
DECODER_START_TOKEN_ID = PAD_IDX


def _shift_right(target_ids):
    '''
    Build decoder_input by shifting target_ids right and prepending the decoder start
    token. decoder_targets is the original target sequence (the token the decoder should
    predict at each step).

        target     = [t0, t1, ..., tN-1, </s>]
        dec_input  = [START, t0, t1, ..., tN-1]
        dec_target = [t0,    t1, ..., tN-1, </s>]
    '''
    start = torch.tensor([DECODER_START_TOKEN_ID], dtype=target_ids.dtype)
    decoder_input = torch.cat([start, target_ids[:-1]], dim=0)
    decoder_target = target_ids
    return decoder_input, decoder_target


def normal_collate_fn(batch):
    '''
    Dynamic padding for train / dev batches.

    Returns:
        * encoder_ids:    BxT  padded encoder input ids
        * encoder_mask:   BxT  1 for real tokens, 0 for padding
        * decoder_inputs: BxT' padded decoder input ids (shifted right)
        * decoder_targets:BxT' padded decoder target ids (PAD_IDX in padded positions,
                               which the training loss masks out)
        * initial_decoder_inputs: Bx1 the first decoder input token (start token) for
                               generation during evaluation
    '''
    encoder_seqs = [ex['encoder_ids'] for ex in batch]
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_inputs_list = []
    decoder_targets_list = []
    for ex in batch:
        dec_in, dec_tgt = _shift_right(ex['target_ids'])
        decoder_inputs_list.append(dec_in)
        decoder_targets_list.append(dec_tgt)

    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)

    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full(
        (batch_size, 1), DECODER_START_TOKEN_ID, dtype=torch.long
    )

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Dynamic padding for test inference (no gold SQL).

    Returns:
        * encoder_ids:  BxT padded encoder input ids
        * encoder_mask: BxT padding mask
        * initial_decoder_inputs: Bx1 start token for generation
    '''
    encoder_seqs = [ex['encoder_ids'] for ex in batch]
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full(
        (batch_size, 1), DECODER_START_TOKEN_ID, dtype=torch.long
    )

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
