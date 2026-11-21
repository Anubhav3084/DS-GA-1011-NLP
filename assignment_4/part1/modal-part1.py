from pathlib import Path

import modal

PART1 = Path(__file__).resolve().parent.parent / "part1"

app = modal.App("dsga1011-bert-part1")
outputs = modal.Volume.from_name("dsga1011-bert-part1-outputs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(PART1 / "requirements.txt")
    .env({"NLTK_DATA": "/opt/nltk_data"})
    .run_commands("python -m nltk.downloader -d /opt/nltk_data wordnet punkt")
    .add_local_file(PART1 / "main.py", "/part1/main.py")
    .add_local_file(PART1 / "utils.py", "/part1/utils.py")
)


@app.function(image=image, gpu="A10", cpu=2, memory=8192, timeout=12 * 60 * 60, volumes={"/outputs": outputs})
def train(num_epochs: int = 3, batch_size: int = 8, max_length: int = 512, smoke: bool = False):
    import subprocess
    import sys
    from uuid import uuid4

    if num_epochs < 1 or batch_size < 1 or not 1 <= max_length <= 512:
        raise ValueError("Use positive epochs and batch size, and a max length between 1 and 512.")

    prefix = "smoke" if smoke else "run"
    run_dir = Path("/outputs") / f"{prefix}-{uuid4().hex[:12]}"
    run_dir.mkdir()
    print(f"Part 1 outputs will be saved under {run_dir.name}/", flush=True)

    command = [
        sys.executable, "-u", "/part1/main.py",
        "--num_epochs", str(1 if smoke else num_epochs),
        "--batch_size", str(batch_size),
        "--max_length", str(max_length),
    ]
    stages = (
        [["--debug_train", "--train", "--eval"]]
        if smoke
        else [
            ["--train", "--eval", "--eval_transformed"],
            ["--train_augmented", "--eval", "--eval_transformed"],
        ]
    )
    for flags in stages:
        try:
            subprocess.run(command + flags, cwd=run_dir, check=True)
        finally:
            outputs.commit()
