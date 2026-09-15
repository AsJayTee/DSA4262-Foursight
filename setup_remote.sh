#!/usr/bin/env bash
#
# Set up a fresh VM from your laptop. One command, safe to re-run.
#
#   ./setup_remote.sh ubuntu@123.45.67.89
#   ./setup_remote.sh -i ~/keys/dsa4262.pem ubuntu@123.45.67.89
#
# WINDOWS: run this in Git Bash, not PowerShell.
#
# This runs on YOUR machine. It clones (or updates) the repo on the VM, copies
# your .env across, and runs bootstrap.sh there. Your laptop needs nothing more
# than git, ssh and a filled-in .env.

set -euo pipefail

REPO_URL="https://github.com/TEAM/DSA4262-Foursight.git"
REMOTE_DIR="~/foursight"
KEY=""

usage() {
  echo "Usage: ./setup_remote.sh [-i <keyfile>] user@host"
  echo
  echo "  -i <keyfile>   SSH private key (.pem). Omit if your key is already"
  echo "                 loaded in ssh-agent or configured in ~/.ssh/config."
  exit 1
}

while getopts ":i:h" opt; do
  case "$opt" in
    i) KEY="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

HOST="${1:-}"
if [ -z "$HOST" ]; then
  echo "ERROR: no host given."
  echo
  usage
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
if [ -n "$KEY" ]; then
  if [ ! -f "$KEY" ]; then
    echo "ERROR: key file not found: $KEY"
    exit 1
  fi
  chmod 600 "$KEY" 2>/dev/null || true
  SSH_OPTS+=(-i "$KEY")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: no .env file found at $ENV_FILE"
  echo
  echo "  1. cp .env.example .env"
  echo "  2. Ask in the Telegram groupchat for the shared R2 values."
  echo "  3. Add your own WANDB_API_KEY from https://wandb.ai/authorize"
  echo
  echo "Never commit .env - .gitignore already blocks it."
  exit 1
fi

echo "==> [1/4] Checking connection to $HOST"
if ! ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 "$HOST" "true"; then
  echo
  echo "ERROR: cannot ssh to $HOST."
  echo "  - Is the instance running, and is the IP current?"
  echo "  - If it needs a key file, pass it: ./setup_remote.sh -i <key.pem> $HOST"
  exit 1
fi

echo "==> [2/4] Cloning or updating the repo at $REMOTE_DIR"
# reset --hard rather than a bare pull: a re-run must never leave a stale or
# half-merged checkout behind.
ssh "${SSH_OPTS[@]}" "$HOST" "
  set -eu
  command -v git >/dev/null 2>&1 || { sudo apt-get update -qq && sudo apt-get install -y -qq git; }
  if [ -d $REMOTE_DIR/.git ]; then
    cd $REMOTE_DIR
    git fetch --quiet origin
    git reset --hard --quiet origin/main
    echo '    updated to' \$(git rev-parse --short HEAD)
  else
    git clone --quiet $REPO_URL $REMOTE_DIR
    cd $REMOTE_DIR
    echo '    cloned at' \$(git rev-parse --short HEAD)
  fi
"

echo "==> [3/4] Copying .env"
scp "${SSH_OPTS[@]}" -q "$ENV_FILE" "$HOST:$REMOTE_DIR/.env"
ssh "${SSH_OPTS[@]}" "$HOST" "chmod 600 $REMOTE_DIR/.env"

echo "==> [4/4] Running bootstrap.sh on the VM (this takes a few minutes)"
ssh "${SSH_OPTS[@]}" "$HOST" "cd $REMOTE_DIR && bash bootstrap.sh"

echo
echo "Done. Connect and start working:"
echo
echo "    ssh ${KEY:+-i $KEY }$HOST"
echo "    cd ${REMOTE_DIR/#\~\//} && source .venv/bin/activate"
echo "    make smoke CONFIG=configs/lightgbm.yaml"
echo
echo "Remember to TERMINATE the instance when you are finished."
