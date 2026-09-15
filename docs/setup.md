# Setup

## Read this first: what runs where

This trips people up, so it is at the top.

| | Your laptop | The VM (Ubuntu) |
|---|---|---|
| Needs git, ssh, `.env` | Yes | — |
| Needs Python, the data, `.venv` | **No** | Yes |
| Runs `setup_remote.sh` | **Yes** | No |
| Runs `bootstrap.sh` | **No** | Yes (automatically) |

**`bootstrap.sh` is Ubuntu-only.** It uses `apt-get`. Running it on a Windows
or macOS laptop produces a wall of errors. You never need to run it by hand —
`setup_remote.sh` runs it on the VM for you.

**On Windows, run `setup_remote.sh` in Git Bash, not PowerShell.** It is a bash
script. Git Bash ships with Git for Windows; open it from the Start menu, or
right-click in the repo folder and choose "Git Bash Here".

`make` is also not available in Git Bash. On Windows, call the scripts directly
instead:

```bash
python scripts/doctor.py
python scripts/train.py --config configs/lightgbm.yaml --smoke
```

---

## One-time: your `.env`

```bash
cp .env.example .env
```

Then fill it in:

1. **The shared R2 block** — ask in the Telegram groupchat. These are the same
   for everyone.
2. **`WANDB_API_KEY`** — *yours*, from <https://wandb.ai/authorize>. Do not use
   someone else's: per-person attribution on the shared W&B project is how we
   each show our own contribution.

Never commit `.env`. `.gitignore` already blocks it.

---

## Spinning up a VM

From your laptop, in the repo folder:

```bash
./setup_remote.sh ubuntu@<instance-ip>
```

If the instance needs a key file:

```bash
./setup_remote.sh -i ~/keys/dsa4262.pem ubuntu@<instance-ip>
```

This clones or updates the repo on the VM, copies your `.env` across, installs
everything, downloads the data from R2, and finishes by running `make doctor`.
It takes a few minutes the first time and is safe to re-run.

Then:

```bash
ssh ubuntu@<instance-ip>
cd foursight && source .venv/bin/activate
make smoke CONFIG=configs/lightgbm.yaml
```

### When something goes wrong

Run `make doctor` (or `python scripts/doctor.py`). It checks every part of the
setup and names the command that fixes whatever is missing. Start there before
reading tracebacks.

---

## Instance types

Task 1 needs very little. The full training set is 180 MB compressed, a full
parse takes about 25 seconds, and a five-fold LightGBM run finishes in well
under two minutes on a small instance.

| Task | Instance | Roughly |
|---|---|---|
| Task 1 — training, experiments | `t3.large` (2 vCPU, 8 GB) | $0.08/hr |
| Task 1 — heavy CV sweeps | `t3.xlarge` / `c6i.2xlarge` | $0.17–0.34/hr |
| Task 2 — SG-NEx batch prediction | `c6i.2xlarge` + more disk | $0.34/hr |

**Do not spin up a GPU instance.** Nothing here needs one, and AWS overspending
is an explicit 5% deduction on the project grade. If a MIL architecture later
turns out to need a GPU, raise it with the team first and update this table.

**Terminate instances when you finish — not "stop".** A stopped instance still
bills for its disk.

You do not need to pull results off an instance before terminating it: training
runs log metrics and the model itself to W&B. Do make sure your **code** is
pushed, though — W&B does not back that up.

---

## Local development without a VM

You can run everything locally if you already have the data:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[train,dev]"
```

Point at data you already have, rather than re-downloading it, by setting this
in `.env`:

```
M6A_DATA_DIR=data0
```

Then:

```bash
python scripts/train.py --config configs/lightgbm.yaml --smoke
python -m pytest -q
```

The test suite builds its own synthetic dataset, so it runs with no data
present at all.
