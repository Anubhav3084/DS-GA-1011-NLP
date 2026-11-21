from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent

app = modal.App("dsga1011-t5-training")
outputs = modal.Volume.from_name("dsga1011-t5-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(ROOT / "requirements.txt")
    .add_local_dir(ROOT / "data", "/project/data")
    .add_local_file(ROOT / "records/ground_truth_dev.pkl", "/project/ground_truth_dev.pkl")
    .add_local_file(ROOT / "train_t5.py", "/project/train_t5.py")
    .add_local_file(ROOT / "t5_utils.py", "/project/t5_utils.py")
    .add_local_file(ROOT / "load_data.py", "/project/load_data.py")
    .add_local_file(ROOT / "utils.py", "/project/utils.py")
)


@app.function(image=image, gpu="A10", timeout=12 * 60 * 60, volumes={"/outputs": outputs})
def train():
    import shutil
    import subprocess
    import sys
    from tempfile import TemporaryDirectory

    project = Path("/project")
    with TemporaryDirectory() as work_dir:
        workspace = Path(work_dir)
        (workspace / "data").symlink_to(project / "data", target_is_directory=True)
        for name in ("checkpoints", "results", "records"):
            destination = Path("/outputs") / name
            destination.mkdir(parents=True, exist_ok=True)
            (workspace / name).symlink_to(destination, target_is_directory=True)
        shutil.copyfile(project / "ground_truth_dev.pkl", workspace / "records/ground_truth_dev.pkl")

        try:
            subprocess.run(
                [
                    sys.executable, "-u", str(project / "train_t5.py"),
                    "--finetune",
                    "--experiment_name", "experiment",
                    "--learning_rate", "3e-4",
                    "--weight_decay", "0.01",
                    "--scheduler_type", "linear",
                    "--num_warmup_epochs", "1",
                    "--max_n_epochs", "20",
                    "--patience_epochs", "5",
                    "--batch_size", "16",
                    "--test_batch_size", "16",
                ],
                cwd=workspace,
                check=True,
            )
        finally:
            outputs.commit()
