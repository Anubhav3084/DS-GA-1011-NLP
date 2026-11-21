import os
import argparse

import torch
from transformers import GenerationConfig, T5ForConditionalGeneration

from load_data import get_dataloader, T5_CHECKPOINT
from utils import save_queries_and_records, set_random_seeds, compute_metrics
from train_t5 import tokenizer  # shared T5 tokenizer

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
PAD_IDX = 0


def get_args():
    ap = argparse.ArgumentParser(description='Generate Part 2 test submission files')
    ap.add_argument('--finetune', action='store_true', default=True,
                    help='Whether the checkpoint is a finetuned model (affects naming: ft vs scr)')
    ap.add_argument('--scratch', action='store_true',
                    help='Set if the checkpoint was trained from scratch (model_type=scr)')
    ap.add_argument('--experiment_name', type=str, default='experiment')
    ap.add_argument('--checkpoint_dir', type=str, default=None,
                    help='Explicit checkpoint dir; defaults to '
                         'checkpoints/<ft|scr>_experiments/<experiment_name>/<best|last>')
    ap.add_argument('--best', dest='best', action='store_true', default=True)
    ap.add_argument('--last', dest='best', action='store_false')
    ap.add_argument('--test_batch_size', type=int, default=32)
    ap.add_argument('--num_beams', type=int, default=1)
    ap.add_argument('--max_gen_length', type=int, default=512)
    ap.add_argument('--also_dev', action='store_true',
                    help='Also regenerate dev predictions + print dev metrics (useful for Q6).')
    return ap.parse_args()


def load_checkpoint(args):
    model_type = 'scr' if args.scratch else 'ft'
    if args.checkpoint_dir is not None:
        load_path = args.checkpoint_dir
    else:
        subdir = 'best' if args.best else 'last'
        load_path = os.path.join('checkpoints', f'{model_type}_experiments',
                                 args.experiment_name, subdir)
    if not os.path.isdir(load_path):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {load_path}\n"
            f"Train the model first (train_t5.py) or pass --checkpoint_dir explicitly."
        )
    print(f"Loading checkpoint from {load_path}")
    model = T5ForConditionalGeneration.from_pretrained(load_path).to(DEVICE)
    model.eval()
    return model, model_type


def generate(model, loader, has_targets, num_beams, max_gen_length):
    gen_config = GenerationConfig(
        max_length=max_gen_length, num_beams=num_beams,
        early_stopping=True, pad_token_id=PAD_IDX,
    )
    out = []
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(loader):
            if has_targets:
                encoder_input, encoder_mask = batch[0], batch[1]
            else:
                encoder_input, encoder_mask = batch[0], batch[1]
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            gen_ids = model.generate(input_ids=encoder_input,
                                     attention_mask=encoder_mask,
                                     generation_config=gen_config)
            out.extend([s.strip() for s in tokenizer.batch_decode(gen_ids, skip_special_tokens=True)])
    return out


def main():
    args = get_args()
    set_random_seeds(42)
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)

    model, model_type = load_checkpoint(args)
    exp = args.experiment_name

    # ---- TEST set (required submission files) ----
    test_loader = get_dataloader(args.test_batch_size, 'test')
    n_test = len(test_loader.dataset)
    test_sql = generate(model, test_loader, has_targets=False,
                        num_beams=args.num_beams, max_gen_length=args.max_gen_length)
    assert len(test_sql) == n_test, (
        f"Prediction count ({len(test_sql)}) != number of test NL queries ({n_test}); "
        f"ordering/length must match data/test.nl exactly.")

    test_sql_path = f'results/t5_{model_type}_{exp}_test.sql'
    test_rec_path = f'records/t5_{model_type}_{exp}_test.pkl'
    save_queries_and_records(test_sql, test_sql_path, test_rec_path)
    print(f"\n[OK] Wrote {len(test_sql)} test predictions:")
    print(f"     {test_sql_path}")
    print(f"     {test_rec_path}")

    # ---- Optional: dev predictions for error analysis / metrics ----
    if args.also_dev:
        dev_loader = get_dataloader(args.test_batch_size, 'dev')
        dev_sql = generate(model, dev_loader, has_targets=True,
                           num_beams=args.num_beams, max_gen_length=args.max_gen_length)
        dev_sql_path = f'results/t5_{model_type}_{exp}_dev.sql'
        dev_rec_path = f'records/t5_{model_type}_{exp}_dev.pkl'
        save_queries_and_records(dev_sql, dev_sql_path, dev_rec_path)
        sql_em, record_em, record_f1, errs = compute_metrics(
            'data/dev.sql', dev_sql_path, 'records/ground_truth_dev.pkl', dev_rec_path)
        err_rate = sum(1 for m in errs if m) / max(len(errs), 1)
        print(f"\n[DEV] Record F1: {record_f1*100:.2f} | Record EM: {record_em*100:.2f}% | "
              f"SQL EM: {sql_em*100:.2f}% | SQL error rate: {err_rate*100:.2f}%")


if __name__ == '__main__':
    main()
