#!/usr/bin/env sh
# Run a batch of comparisons end to end, locally or on Ronin.
#
# This drives scripts/evaluate.py; it computes nothing itself and it is not
# shared infrastructure. Every line below is a command you could type by hand -
# the script exists so that a whole batch survives one arm failing, and so that
# a Ronin instance can be handed one command and left alone.
#
#   sh analysis/evaluation/run_batch.sh            # everything not yet done
#   sh analysis/evaluation/run_batch.sh groupA     # just one group
#   sh analysis/evaluation/run_batch.sh --list     # print the plan, run nothing
#
# Works in Git Bash on Windows and on a Ronin Ubuntu box: it finds the venv
# interpreter either way. Set M6A_DATA_DIR if the data is not in data/raw.
#
# WHY IT DOES NOT STOP ON FAILURE. Ronin instances are terminated with nothing
# pulled off them (docs/decisions/0007), so a batch that dies on arm 2 of 6 and
# sits idle until someone notices has wasted the instance. Each arm is logged,
# failures are counted and named at the end, and the rest keep going. Every
# result is in W&B before the next arm starts.

set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
else
    PY=$(command -v python3 || command -v python)
fi
[ -n "${PY:-}" ] || { echo "No python interpreter found. Create .venv first."; exit 1; }

LOGS="$ROOT/analysis/evaluation/logs"
mkdir -p "$LOGS"

# --------------------------------------------------------------------------
# The plan. One line per arm:  group | label | arguments to evaluate.py
#
# Ordered cheap-and-decisive first. Group C shares one hypothesis and the three
# arms attribute it to a structure, so run all three or none - a single one
# cannot say which graph is worth building.
# --------------------------------------------------------------------------
PLAN=$(cat <<'PLAN'
groupA|flank_x_depth|--config configs/quantiles_flank_depth_augmented.yaml --compare-features quantiles_v1
groupA|flank_vs_depthaug|--config configs/quantiles_flank_depth_augmented.yaml --compare-run psdwqobf
groupA|flank_ablation|--config configs/quantiles_flank.yaml --ablate
groupC|nbr_struct|--config configs/nbr_struct.yaml --compare-features quantiles_v1
groupC|nbr_signal|--config configs/nbr_signal.yaml --compare-features quantiles_v1
groupC|transcript|--config configs/transcript.yaml --compare-features quantiles_v1
groupB|coupling|--config configs/coupling.yaml --compare-features quantiles_v1
PLAN
)

WANT=${1:-all}

if [ "$WANT" = "--list" ]; then
    echo "$PLAN" | while IFS='|' read -r group label args; do
        printf '  %-8s %-20s %s\n' "$group" "$label" "$args"
    done
    exit 0
fi

STARTED=$(date +%s)
FAILED=""
COUNT=0

echo "$PLAN" | {
    while IFS='|' read -r group label args; do
        [ "$WANT" = "all" ] || [ "$WANT" = "$group" ] || continue
        COUNT=$((COUNT + 1))
        LOG="$LOGS/${label}.log"
        echo "=============================================================="
        echo "[$group] $label   ->  $LOG"
        echo "  $PY scripts/evaluate.py $args"
        echo "  started $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        ARM_START=$(date +%s)
        # shellcheck disable=SC2086
        if "$PY" scripts/evaluate.py $args >"$LOG" 2>&1; then
            echo "  OK in $((($(date +%s) - ARM_START) / 60)) min"
            grep -E "corrected \(quote this\)|is \+?-?0\.[0-9]+ (better|worse)" "$LOG" | tail -2
        else
            echo "  FAILED after $((($(date +%s) - ARM_START) / 60)) min - see $LOG"
            tail -15 "$LOG"
            FAILED="$FAILED $label"
        fi
    done

    echo "=============================================================="
    echo "batch done in $((($(date +%s) - STARTED) / 60)) min, $COUNT arm(s)"
    if [ -n "$FAILED" ]; then
        echo "FAILED:$FAILED"
        echo "Everything else is in W&B. Re-run only the failures."
        exit 1
    fi
    echo "All arms logged to W&B. Pull results with scripts/compare_runs.py."
}
