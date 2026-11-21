# DS-GA 1011 Assignment 4: Fine-tuning Language Models

Starter code for Assignment 4. See the assignment writeup (`main.tex`/`main.pdf` in the course
handout) for the full problem statement.

Part 1 (BERT fine-tuning for sentiment classification) and Part 2 (T5 fine-tuning for text-to-SQL)
pin different, incompatible package versions, so set up a **separate environment for each part** —
see `part1/README.md` and `part2/README.md` for setup instructions specific to each part.

You will require GPUs for this homework. You can either use the HPC server (see documentation
[here](https://github.com/Athul-R/NYU-HPC-Tutorials)) or use an API service like
[lightning.ai](https://lightning.ai/) or other cloud providers.

## Layout

- `part1/`: starter code for Part 1 (`main.py`, `utils.py`). You will modify `main.py` and `utils.py`.
- `part2/`: starter code for Part 2 (`train_t5.py`, `t5_utils.py`, `utils.py`, `load_data.py`,
  `evaluate.py`), the text-to-SQL `data/`, the ground-truth dev records in `records/`, and a
  `results/` directory where you should save your model's output SQL/records for submission.
