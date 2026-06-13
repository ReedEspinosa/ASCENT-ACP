#!/bin/bash
# 2021-only rerun after the sub-second-CDP merge fix (2020 already complete).
set -u
cd "$(dirname "$0")"

DATA=/Users/wrespino/Synced/ACMAP_Meloe/SuborbitalDataSets/ACTIVATE
LOGDIR=$DATA/processing_logs
mkdir -p "$LOGDIR" "$DATA/isara_output"

echo "===== STAGE 1: ICARTT merge 2021 $(date) ====="
python run_full_activate_merge.py --years 2021 --n-workers 6 \
  > "$LOGDIR/merge_2021.log" 2>&1
echo "merge 2021 exit code: $?"

PKL=$DATA/merged1sec_allInstruments_2021_V2.pkl
if [ ! -f "$PKL" ]; then
  echo "!! $PKL missing after merge; aborting 2021"
  echo "===== 2021 ABORTED $(date) ====="
  exit 1
fi

echo "===== STAGE 2: clock alignment 2021 $(date) ====="
python apply_clock_alignment.py "$PKL" > "$LOGDIR/clock_align_2021.log" 2>&1
RC=$?
echo "clock alignment 2021 exit code: $RC"
if [ $RC -ne 0 ]; then
  echo "!! clock alignment failed for 2021; aborting"
  exit 1
fi

echo "===== STAGE 3: ISARA pipeline 2021 $(date) ====="
python -m ASCENT_ACP.pipeline --config configs/activate_2021_full.json --plots \
  > "$LOGDIR/isara_2021.log" 2>&1
echo "ISARA 2021 exit code: $?"

echo "===== 2021 DONE $(date) ====="
