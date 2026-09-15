#!/usr/bin/env bash
#
# Set up this machine to run experiments. Idempotent - safe to re-run.
#
# THIS RUNS ON THE VM (Ubuntu). It uses apt-get, so running it on a Windows or
# macOS laptop will fail. Your laptop does not need it: see docs/setup.md.
#
# Normally you do not run this by hand - setup_remote.sh does it for you.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "--> System packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip build-essential
else
  echo "    apt-get not found - skipping. This script targets Ubuntu."
  echo "    If you are on a laptop, you do not need it. See docs/setup.md."
fi

echo "--> Python environment (.venv)"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
echo "--> Installing m6a and dependencies"
pip install --quiet -e ".[train,dev]"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "--> Logging in to Weights & Biases"
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 \
    && echo "    logged in as ${WANDB_ENTITY:-personal}" \
    || echo "    WARNING: wandb login failed - training still works, tracking will not"
else
  echo "--> No WANDB_API_KEY in .env - skipping W&B login"
fi

echo "--> Downloading data from R2"
if python scripts/download_data.py --set course; then
  :
else
  echo "    WARNING: download failed. Check R2 keys in .env, then re-run:"
  echo "             python scripts/download_data.py"
fi

echo
echo "================ make doctor ================"
python scripts/doctor.py || true
