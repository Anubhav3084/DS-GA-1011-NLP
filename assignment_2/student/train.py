import sys
import torch
from modules import *
from data_utils import *
from pathlib import Path

root = Path.cwd().parent          # assignment/
sys.path.insert(0, str(root))

data = load_memmap_dataset('/Users/anubhav/Desktop/Courses/Fall 26/Assignments/NLP/assignment_2/data/tinystories_valid.bin', '/Users/anubhav/Desktop/Courses/Fall 26/Assignments/NLP/assignment_2/data/tinystories_valid.meta')
vocab, merges = load_vocab("/Users/anubhav/Desktop/Courses/Fall 26/Assignments/NLP/assignment_2/data/tinystories_vocab_merges.pt")

params = {
    "VOCAB_SIZE": 10_000,
    "CONTEXT_LENGTH": 128,
    "D_MODEL": 128,
    "NUM_LAYERS": 4,
    "NUM_HEADS": 4,
    "D_FF": 384,
    "ROPE_THETA": 10000.0,
    "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    
    "BATCH_SIZE": 32,
    "MAX_LR": 1e-3,
    "MIN_LR": 1e-4,
    "WARMUP_ITERS": 100,
    "TOTAL_STEPS": 500,
    "WEIGHT_DECAY": 0.01,
    "MAX_GRAD_NORM": 1.0
}

if __name__ == "__main__":

    MyGPT = TransformerLM(
        vocab_size=params["VOCAB_SIZE"],
        context_length=params["CONTEXT_LENGTH"],
        d_model=params["D_MODEL"],
        num_layers=params["NUM_LAYERS"],
        num_heads=params["NUM_HEADS"],
        d_ff=params["D_FF"],
        rope_theta=params["ROPE_THETA"],
        device=params["DEVICE"]
    )

    MyOptimizer = AdamW(params=MyGPT.parameters())

    for k,v in params.items():
        print(f"{k}: {v}")

    losses = []
    MyGPT.train()
    for step in range(params['TOTAL_STEPS']):
        # 1. set the scheduled lr
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=params['MAX_LR'],
            min_learning_rate=params['MIN_LR'],
            warmup_iters=params['WARMUP_ITERS'],
            cosine_cycle_iters=params['TOTAL_STEPS']
        )
        for group in MyOptimizer.param_groups:
            group["lr"] = lr

        # 2. sample a batch
        x, y = get_batch(data, params['BATCH_SIZE'], params['CONTEXT_LENGTH'], params['DEVICE'])

        # 3. forward, loss, backward
        logits = MyGPT(x)                      # (batch, seq, vocab)
        loss = cross_entropy(inputs=logits, targets=y)        # your version handles the extra dim
        MyOptimizer.zero_grad()
        loss.backward()

        # 4. clip, then step
        gradient_clipping(parameters=MyGPT.parameters(), max_l2_norm=params['MAX_GRAD_NORM'])
        MyOptimizer.step()

        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")