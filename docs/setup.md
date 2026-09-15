# Setup

Getting a machine ready to run experiments. For *how to run an experiment* once
it is ready, see [running-experiments.md](running-experiments.md).

---

## Read this first: what runs where

This is the thing that trips people up.

| | Your laptop | The Ronin instance |
|---|---|---|
| Needs git, ssh, your `.env`, your `.pem` key | Yes | — |
| Needs Python, the data, `.venv` | **No** | Yes |
| Runs `setup_remote.sh` | **Yes** | No |
| Runs `bootstrap.sh` | **No** | Yes, automatically |

**`bootstrap.sh` is Ubuntu-only.** It uses `apt-get`. Running it on a Windows
or macOS laptop produces a wall of errors. You never run it by hand.

**On Windows, use Git Bash — not PowerShell, not CMD.** `setup_remote.sh` is a
bash script. Git Bash ships with Git for Windows: find it in the Start menu, or
right-click inside the repo folder and choose "Git Bash Here".

`make` does not exist in Git Bash either. On your laptop, call the scripts
directly instead:

```bash
python scripts/doctor.py
python scripts/train.py --config configs/lightgbm.yaml --smoke
```

---

## One-time setup on your laptop

### 1. Clone the repo

```bash
git clone https://github.com/AsJayTee/DSA4262-Foursight.git
cd DSA4262-Foursight
```

### 2. Create your `.env`

```bash
cp .env.example .env
```

Then fill in two things:

- **The shared R2 block** — ask in the Telegram groupchat. Identical for everyone.
- **`WANDB_API_KEY`** — *yours*, from <https://wandb.ai/authorize>. Do not use
  someone else's. Per-person attribution on the shared W&B project is how each
  of us evidences our own contribution for the individual grade.

Never commit `.env`. `.gitignore` already blocks it, and nothing should ever
change that.

### 3. Put your Ronin key somewhere sane, with the right permissions

Download the `.pem` key file from Ronin and move it into your `.ssh` folder:

```bash
mkdir -p ~/.ssh
mv ~/Downloads/yourkey.pem ~/.ssh/
chmod 600 ~/.ssh/yourkey.pem
```

**`chmod 600` matters.** SSH refuses to use a private key that other users can
read, and the error it gives ("UNPROTECTED PRIVATE KEY FILE") does not make the
fix obvious. In Git Bash, `~` is `C:\Users\<your name>`.

#### If you use `~/.ssh/config`

A config entry saves typing, but the `IdentityFile` path **must start with `~/`**:

```
Host jinthautest.nus.cloud
  HostName jinthautest.nus.cloud
  IdentityFile ~/.ssh/yourkey.pem     # NOT .ssh/yourkey.pem
  User ubuntu
```

A relative path like `.ssh/yourkey.pem` is resolved against your *current
directory*, not your home directory. It works when you happen to be sitting in
`~`, and silently fails from the repo folder — which is where you will actually
be running things.

---

## Running an experiment machine on Ronin

### Step 1 — launch the instance

Launch an Ubuntu instance in Ronin and note its hostname or IP. The smallest
tier is plenty: **2 vCPU / 8 GB**. Training on the full dataset takes about 90
seconds.

**Do not launch a GPU instance.** Nothing here needs one, and AWS overspending
is an explicit 5% deduction on the project grade.

### Step 2 — from your laptop, in Git Bash

From inside the repo folder:

```bash
./setup_remote.sh -i ~/.ssh/yourkey.pem ubuntu@<your-instance-host>
```

If your `~/.ssh/config` already has an entry for the host, the short form works:

```bash
./setup_remote.sh <your-instance-host>
```

This is the only command you need. It clones or updates the repo on the
instance, copies your `.env` across, installs Python and everything else,
downloads the data from R2, logs into W&B, and finishes by running `doctor`.

**It is safe to run again, any time.** If something looks wrong, re-running is
the first thing to try — it resets the instance to a known-good state and
skips work already done.

Expected output:

```
==> [1/4] Checking connection to ubuntu@...
==> [2/4] Cloning or updating the repo at ~/foursight
    cloned at 2ad59c1
==> [3/4] Copying .env
==> [4/4] Running bootstrap.sh on the VM (this takes a few minutes)
--> System packages
--> Python environment (.venv)
--> Installing m6a and dependencies
--> Logging in to Weights & Biases
    logged in as dsa4262-team
--> Downloading data from R2
  2 object(s) for --set course -> data/raw/
  [downloaded] data.info.labelled  (4.7 MB)
  [downloaded] dataset0.json.gz  (179.6 MB)

================ make doctor ================
[  OK  ] python 3.14.4
...
Ready.
```

First run takes a few minutes, mostly pip. A second run finishes in under a
minute, with the data reported as `skipped (size match)`.

### Step 3 — in the SSH terminal

```bash
ssh -i ~/.ssh/yourkey.pem ubuntu@<your-instance-host>
```

Then, every time you connect:

```bash
cd ~/foursight
source .venv/bin/activate
```

**That `source` line is required.** Without it `python` is the system Python and
nothing will import. If you get `ModuleNotFoundError: No module named 'm6a'`,
this is why.

Now you can run:

```bash
make doctor                                  # is this machine healthy?
make smoke CONFIG=configs/lightgbm.yaml      # ~11s, 5,000 sites, no W&B
make train CONFIG=configs/lightgbm.yaml      # ~90s, full data, logs to W&B
make test                                    # the test suite
make predict INPUT=<file.json.gz> OUTPUT=<out.csv>
```

To pick up code your teammates have pushed since you connected:

```bash
git pull
```

You do not need to reinstall after a `git pull` — the package is installed in
editable mode, so code changes take effect immediately.

### Step 4 — when you are done

**Terminate the instance in Ronin. Not "stop".** A stopped instance still bills
for its disk.

You do not need to copy anything off it first: `make train` logs metrics *and*
the trained model to W&B, so results survive termination.

**Do make sure your code is pushed**, though — W&B does not back that up.

```bash
git add -A && git commit -m "what you did" && git push
```

---

## When something goes wrong

Run this first:

```bash
make doctor
```

It checks every part of the setup and names the command that fixes whatever is
missing. Read it before reading a traceback.

| Symptom | Cause |
|---|---|
| `cannot ssh to <host>` | Key not passed — add `-i ~/.ssh/yourkey.pem` |
| `UNPROTECTED PRIVATE KEY FILE` | `chmod 600 ~/.ssh/yourkey.pem` |
| `no .env file found` | You skipped step 2 — ask on Telegram for the shared values |
| `ModuleNotFoundError: No module named 'm6a'` | You forgot `source .venv/bin/activate` |
| `No objects found for --set course` | R2 keys wrong or expired — check `.env` |
| `bootstrap.sh` errors on apt | Not an Ubuntu instance |
| `make: command not found` | You are on your laptop, not the instance — call `python scripts/...` directly |

If `setup_remote.sh` itself fails, **say so rather than fixing that instance by
hand**. A failure there will hit everyone, so it should be fixed in the script.

---

## Local development without an instance

You can run everything on your own machine if you want:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv/Scripts/activate
pip install -e ".[train,dev]"
python scripts/download_data.py
python scripts/train.py --config configs/lightgbm.yaml --smoke
python -m pytest -q
```

If you already have the data somewhere, point at it instead of re-downloading
by setting this in `.env`:

```
M6A_DATA_DIR=data0
```

The test suite builds its own synthetic dataset, so `pytest` works with no data
present at all.

---

## Instance sizing and cost

| Task | Instance | Roughly |
|---|---|---|
| Training, experiments (Task 1) | 2 vCPU / 8 GB | $0.08/hr |
| Heavy CV sweeps | 4–8 vCPU | $0.17–0.34/hr |
| SG-NEx batch prediction (Task 2) | 8 vCPU + more disk | $0.34/hr |

Each student gets US$100 of credit, pooled at team level — any one person can
spend all of it. Overspending costs the team 5% of the project grade.

Raise it with the team before launching anything larger than the table above.
