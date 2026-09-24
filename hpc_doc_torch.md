# NYU Torch HPC Tutorial

- Official documentation: [https://services.rt.nyu.edu/docs/hpc/getting_started/intro/](https://services.rt.nyu.edu/docs/hpc/getting_started/intro/)
- Chatbot: [https://chatbot-torch.apps.cloud.rt.nyu.edu/](https://chatbot-torch.apps.cloud.rt.nyu.edu/)

## Contents

- [1. Access](#1-access)
    - [1.1 Slurm Account and Resource Allocation](#11-slurm-account-and-resource-allocation)
    - [1.2 Option A — SSH](#12-option-a-ssh)
    - [1.3 Option B — Open OnDemand (browser)](#13-option-b-open-ondemand-browser)
- [2. Storage and Data Transfers](#2-storage-and-data-transfers)
    - [2.1 Which filesystem to use](#21-which-filesystem-to-use)
    - [2.2 Transferring files](#22-transferring-files)
- [3. Apptainer Setup](#3-apptainer-setup)
    - [3.0 Concepts: image, container, overlay](#30-concepts-image-container-overlay)
    - [3.1 Get an interactive node](#31-get-an-interactive-node)
    - [3.2 Create the overlay](#32-create-the-overlay)
    - [3.3 Choose a base image](#33-choose-a-base-image)
    - [3.4 Build the environment with uv](#34-build-the-environment-with-uv)
    - [3.5 Verify on a GPU node](#35-verify-on-a-gpu-node)
- [4. Running Jobs with Slurm](#4-running-jobs-with-slurm)
    - [4.1 Interactive vs batch](#41-interactive-vs-batch)
    - [4.2 Requesting resources](#42-requesting-resources)
    - [4.3 Example `sbatch` script](#43-example-sbatch-script)
    - [4.4 Submitting and monitoring](#44-submitting-and-monitoring)
- [Troubleshooting](#troubleshooting)

---

## 1. Access

### 1.1 Slurm Account and Resource Allocation

Every job on Torch must be charged to a **Slurm account**. Class enrollment was only just finalized
and the allocation request for this course is still in progress.

|  | Status |
| --- | --- |
| **Slurm account** | *Pending* |
| **GPU hours** | *Pending* |
| **CPU hours** | *Pending* |
| **GPU types** | *Pending* — Torch has L40S (48 GB), H200 (141 GB), H100, A100 |

> **We will announce these as soon as the allocation is approved.**
<!--
> Until then you can log in,
move files, and build your environment (Sections 2–3), but you cannot submit jobs.
>

**Two separate things are required, and having the first does not give you the second:**

|  | What it gives you | How you get it |
| --- | --- | --- |
| **1. HPC account** | The ability to *log in* to Torch | Requested **for the whole class by the instructor**; see below |
| **2. Slurm account** | The ability to *run jobs* | From a course allocation a PI creates in the HPC Project Portal |

> *“An HPC account gives you access to Torch, but an active allocation within the HPC projects
management portal gives you access to a SLURM account which is needed to run jobs on Torch.”*
>

**HPC accounts for this course are requested in bulk by the course staff** — you do not need to
apply individually. Course accounts are valid **until the end of the semester**.

If you need an account outside the course roster (e.g. you joined late), you can request one
yourself at [https://identity.it.nyu.edu/](https://identity.it.nyu.edu/) — **NYU VPN required** — under
**Manage Accounts → Request HPC Account**. The form requires a **faculty sponsor**, a reason for
the request, and consent to the terms of use. Your sponsor is then notified and
*“provisioning will only occur after approval”*, so this route is not instant.
[Full instructions.](https://services.rt.nyu.edu/docs/hpc/getting_started/HPC_Accounts/getting_and_renewing_an_account/)
-->

Once the account is announced, check that you are on it:

```bash
my_slurm_accounts
```

### 1.2 Option A — SSH

[NYU VPN](https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/#connecting-to-the-nyu-network) is required off campus.

```bash
ssh [netid]@login.torch.hpc.nyu.edu
```

Torch uses **Microsoft device-code authentication**:

```
Authenticate with PIN XXXXXXXXX at https://login.microsoft.com/device and press ENTER.
```

- Tip: [Configure Your SSH Client](https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/#configuring-your-ssh-client)

### 1.3 Option B — Open OnDemand (browser)

[https://ood.torch.hpc.nyu.edu](https://ood.torch.hpc.nyu.edu/) — same VPN requirement, no local SSH client needed.

- Documentation: [https://services.rt.nyu.edu/docs/hpc/ood/ood_intro/](https://services.rt.nyu.edu/docs/hpc/ood/ood_intro/)

| Menu | Runs on | Notes |
| --- | --- | --- |
| **Clusters → Torch Shell Access** | **login node — no GPU** | Equivalent to `ssh`, in a browser |
| **Interactive Apps → Jupyter** | **compute node — can have a GPU** | Submits a Slurm job for you |
| **Files** | — | Drag-and-drop upload/download; preview without downloading |
| **Jobs → Active Jobs** | — | A graphical `squeue` |

---

## 2. Storage and Data Transfers

### 2.1 Which filesystem to use

[Full comparison table](https://services.rt.nyu.edu/docs/hpc/storage/intro_and_data_management/#hpc-storage-comparison-table):

| Space | Variable | Purpose | Backed up / Flushed | Quota (space / files) |
| --- | --- | --- | --- | --- |
| `/home` | `$HOME` | “best for small files” | **YES** / NO | 50 GB / **30 K** |
| `/scratch` | `$SCRATCH` | “Best for large files” | **NO** / not accessed in 60 days | 5 TB / 5 M |
| `/archive` | `$ARCHIVE` | Long-term storage | YES / NO | 2 TB / 20 K |
- Working directory: `/scratch`
- Check your usage at any time: `myquota`

### 2.2 Transferring files

Use the **data transfer node** `dtn.torch.hpc.nyu.edu`, **not** the login node

> Transferring on login nodes is a bad practice because it can degrade the node’s performance.
>

```bash
# laptop -> Torch
rsync -avP ./my_project/ [netid]@dtn.torch.hpc.nyu.edu:/scratch/[netid]/my_project/

# Torch -> laptop
rsync -avP [netid]@dtn.torch.hpc.nyu.edu:/scratch/[netid]/results/ ./results/
```

- `a` preserves metadata, `v` lists what moved, `P` shows progress and lets an interrupted
transfer resume. `scp` works the same way but cannot resume.

> **Tip: send the HuggingFace cache to scratch.** It defaults to `~/.cache/huggingface` in `/home`,
which has only 30 K inodes — one mid-sized model exhausts it. Add to `~/.bashrc`:
>
>
> ```bash
> export HF_HOME=/scratch/$USER/.cache/huggingface
> ```
>

---

## 3. Apptainer Setup

### 3.0 Concepts: image, container, overlay

You have no `sudo` on a cluster, so you cannot `apt install` anything, and the system Python is
whatever the OS shipped. A **container** solves this: you bring your own userland.

**Apptainer** (renamed from Singularity in 2021 — both names appear in NYU’s docs, same tool) is
Docker for HPC. It runs as *you*, not root, and an image is a single file you can `ls`.

Three pieces combine at run time:

```
  ┌─ image (.sif) ──────────────────────────┐   The OS + system libraries. Read-only *at run
  │  e.g. /share/apps/images/<image>.sif    │   time* — which is why you need an overlay.
  └─────────────────────────────────────────┘
                   +
  ┌─ overlay (.ext3) ──────────────────┐   WRITABLE. Appears as /ext3 inside the container.
  │  /scratch/[netid]/containers/...   │   Your uv, Python and torch live here.
  └────────────────────────────────────┘
                   +
  ┌─ --nv ─────────────────────────────┐   Passes the host NVIDIA driver into the container.
  └────────────────────────────────────┘
                   ↓  apptainer exec
            ═══ container ═══              The running process. Exits and disappears.
```

**image vs container** is the same relationship as **executable vs process**: `/bin/ls` sits on disk
permanently; the `ls` process exists only while the command runs. A `.sif` is the file; the
container is what exists during `apptainer exec`. There is no `docker ps` equivalent — when the
command finishes, the container is gone.

**What is `.ext3`?** A filesystem format, like NTFS or APFS. `overlay-15GB-500K.ext3` is *one file*
containing an entire filesystem — the same idea as a `.dmg` or `.iso`. Apptainer mounts it at
`/ext3`, where it becomes a writable disk. The name encodes both limits: **15 GB of space and
500 K inodes**. Those 300,000 PyTorch files count as **one** file to the cluster.

ext3 is a **single-writer** filesystem. Mount the overlay `:rw` while building, `:ro` when running
jobs. Two simultaneous `:rw` mounts will corrupt it.

**The environment lives inside the container. Your code does not.**

```
  Host (Torch itself, outside any container)
  /scratch/[netid]/
      ├── containers/overlay-15GB-500K.ext3   ← the environment (torch, numpy, ...)
      ├── my_project/                         ← YOUR CODE goes here (and in git)
      │     ├── train.py
      │     └── data/
      └── logs/

  Container (exists only during `apptainer exec`)
      /ext3      ← the overlay above, mounted
      /scratch   ← bind-mounted, which is how the container reads your code
```

| Task | Where |
| --- | --- |
| Write / edit / read code | **Outside** the container |
| `git clone` / `pull` / `push` | **Outside** the container |
| Run code | **Inside** the container (`apptainer exec ...`) |

Do **not** put code inside the overlay: it is read-only when jobs run, and git cannot manage it.
Edit code with `code-server` via OpenOnDemand or [VS Code Remote-SSH](https://services.rt.nyu.edu/docs/hpc/tools_and_software/vscode_remote_ssh_torch/),
the OOD Files browser, or plain `git` — all outside the container. The container is not a virtual
machine you live in; it starts, runs one command, and exits.

### 3.1 Get an interactive node

```bash
# CPU node — enough for Sections 3.2-3.4 (building the overlay and environment)
srun --account=[account] --cpus-per-task=8 --mem=24G --time=00:30:00 --pty /bin/bash

# GPU node — needed only to verify CUDA works (Section 3.5)
srun --account=[account] --gres=gpu:1 --constraint='h200|l40s' \
     --cpus-per-task=4 --mem=16G --time=00:30:00 --pty /bin/bash
```

The prompt tells you where you are: `[netid]@torch-login-b-0` is the login node,
`[netid]@cs601` or `[netid]@gl025` is a compute node. Type `exit` when you are done — interactive
allocations are billed by wall-clock, not by how much you use them.

### 3.2 Create the overlay

NYU publishes pre-made overlay templates in several size/inode combinations:

```bash
ls /share/apps/overlay-fs-ext3/
```

For example:

```bash
mkdir -p /scratch/$USER/containers
cd /scratch/$USER/containers
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz          # a few minutes, no progress output
```

- `gunzip` prints nothing while it runs, which looks like a hang. You can check the size/progress by `ls -lh $SCRATCH/containers/overlay-15GB-500K.ext3`

### 3.3 Choose a base image

```bash
ls /share/apps/images/
```

This guide uses **`/share/apps/images/ubuntu-22.04.4.sif`**. Use it directly from the shared
location — no need to copy it into your own space.

`/share/apps/images/` is only a convenient starting point, **not your only option**. You can also [pull images from Docker](https://services.rt.nyu.edu/docs/hpc/tutorial_apptainer/docker_images/) or [build your own image](https://services.rt.nyu.edu/docs/hpc/tutorial_apptainer/customize_container_environments/).

### 3.4 Build the environment with uv

On the CPU node from Section 3.1, launch the container with the overlay **writable**:

```bash
apptainer exec --fakeroot \
  --overlay /scratch/$USER/containers/overlay-15GB-500K.ext3:rw \
  /share/apps/images/ubuntu-22.04.4.sif \
  /bin/bash
```

Inside the container, install `uv` and Python into `/ext3`.

```bash
export UV_INSTALL_DIR=/ext3/uv/bin
export UV_PYTHON_INSTALL_DIR=/ext3/uv/python
export UV_CACHE_DIR=/ext3/uv/cache
export PATH=/ext3/uv/bin:$PATH
mkdir -p /ext3/uv/bin /ext3/uv/python /ext3/uv/cache

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/ext3/uv/bin sh
uv python install 3.11
```

Create the virtual environment and install PyTorch:

```bash
uv venv /ext3/venv --python 3.11
uv pip install --python /ext3/venv torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python /ext3/venv numpy tqdm ipykernel
```

The `cu126` in that URL selects the CUDA 12.6 build of torch. It must be compatible with the **host
driver**, not with the image — check the CUDA version reported by `nvidia-smi` on a GPU node.

Write an activation script so you never repeat any of this:

```bash
cat > /ext3/env.sh <<'EOS'
#!/bin/bash
export UV_PYTHON_INSTALL_DIR=/ext3/uv/python
export UV_CACHE_DIR=/ext3/uv/cache
export VIRTUAL_ENV=/ext3/venv
export PATH=/ext3/uv/bin:/ext3/venv/bin:$PATH
EOS

source /ext3/env.sh
python -c "import torch; print(torch.__version__, torch.version.cuda)"
exit
```

To add packages later: repeat this section, `source /ext3/env.sh`, then `uv pip install ...`.

### 3.5 Verify on a GPU node

```bash
apptainer exec --nv \
  --overlay /scratch/$USER/containers/overlay-15GB-500K.ext3:ro \
  /share/apps/images/ubuntu-22.04.4.sif \
  bash -c 'source /ext3/env.sh; python -c "import torch; print(torch.cuda.is_available())"'
```

Expected output: `True`. Note `--nv` (exposes the driver) and `:ro` (read-only, because this is a
job). `torch.cuda.is_available()` returns `False` on a CPU node even when everything is correct —
that is not a failure.

If `ls /scratch/$USER` is empty inside the container, add `--bind /scratch`.

---

## 4. Running Jobs with Slurm

Reference: [https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)

### 4.1 Interactive vs batch

- **`srun --pty /bin/bash`** — an interactive shell on a compute node. Good for debugging and for
building environments. Dies when your laptop sleeps.
- **`sbatch script.sh`** — queues a job that runs without you. Use this for anything real.

### 4.2 Requesting resources

| Option | Usage |
| --- | --- |
| `--account=[account]` | Required. List yours with `my_slurm_accounts`. |
| `--gres=gpu:N` | Number of GPUs. |
| `--constraint=<type>` | GPU type: `l40s` (48 GB), `h200` (141 GB). `--constraint='h200\|l40s'` accepts either and schedules sooner. |
| `--time=HH:MM:SS` | Wall time. Determines which QoS applies. |
| `--cpus-per-task`, `--mem` | CPU and memory. |
| `--partition` | Do not specify manually (except for preemption). No physical nodes are tied to partitions; resources are allocated via partition QoS. |

QoS limits, from `show_slurm_qos`:

| QoS | Max wall time | Max resources |
| --- | --- | --- |
| interactive | 6 hours | cpu=16, mem=60G |
| cpu_short | 6 hours | cpu=32, mem=120G |
| cpu48 | 2 days | cpu=3000, mem=6000G |
| gpu48 | 2 days | gres/gpu=16 |
| gpu168 | 7 days | gres/gpu=4 |

### 4.3 Example `sbatch` script

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=[account]
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/logs/%j_%x.out
#SBATCH --error=/scratch/%u/logs/%j_%x.err

mkdir -p /scratch/$USER/logs

SIF=/share/apps/images/ubuntu-22.04.4.sif
OVERLAY=/scratch/$USER/containers/overlay-15GB-500K.ext3

apptainer exec --bind /scratch --nv --overlay ${OVERLAY}:ro "$SIF" /bin/bash -c "
  source /ext3/env.sh
  set -euo pipefail
  cd /scratch/$USER/my_project
  python train.py
"
```

Note `:ro` — many jobs can share a read-only overlay, but a second `:rw` mount will crash.

### 4.4 Submitting and monitoring

```bash
sbatch train.sh                  # submit; prints a job ID
squeue -u $USER                  # what am I running?
squeue -u $USER --start          # when will a pending job start?
scancel <jobid>                  # kill it
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES   # post-mortem
tail -f /scratch/$USER/logs/<jobid>_train.out
```

Interactive allocations are billed by **wall-clock, not utilization** — an `srun` shell you forgot
to `exit` bills you all night. Always `exit` when you are done.
