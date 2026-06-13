#!/bin/bash
# Full ACTIVATE pipeline: merge -> clock alignment -> ISARA, both years.
# Logs each stage; continues to the other year if one stage fails.
set -u
cd "$(dirname "$0")"

DATA=/Users/wrespino/Synced/ACMAP_Meloe/SuborbitalDataSets/ACTIVATE
LOGDIR=$DATA/processing_logs
mkdir -p "$LOGDIR" "$DATA/isara_output"

# Seed the optics LUT cache from the test run (identical optics config)
if [ ! -d "$DATA/isara_output/lut_cache" ]; then
  cp -R /Users/wrespino/Synced/ACMAP_Meloe/SuborbitalDataSets/ACTIVATE_TEST/isara_output/lut_cache \
        "$DATA/isara_output/lut_cache"
fi

echo "===== STAGE 1: ICARTT merge (both years) $(date) ====="
python run_full_activate_merge.py --years 2020 2021 --n-workers 6 \
  > "$LOGDIR/merge.log" 2>&1
MERGE_RC=$?
echo "merge exit code: $MERGE_RC"

for Y in 2020 2021; do
  PKL=$DATA/merged1sec_allInstruments_${Y}_V2.pkl
  if [ ! -f "$PKL" ]; then
    echo "!! $PKL missing, skipping year $Y"
    continue
  fi

  echo "===== STAGE 2: clock alignment $Y $(date) ====="
  python apply_clock_alignment.py "$PKL" > "$LOGDIR/clock_align_$Y.log" 2>&1
  RC=$?
  echo "clock alignment $Y exit code: $RC"
  if [ $RC -ne 0 ]; then
    echo "!! clock alignment failed for $Y, skipping ISARA for $Y"
    continue
  fi

  echo "===== STAGE 3: ISARA pipeline $Y $(date) ====="
  python -m ASCENT_ACP.pipeline --config configs/activate_${Y}_full.json --plots \
    > "$LOGDIR/isara_$Y.log" 2>&1
  RC=$?
  echo "ISARA $Y exit code: $RC"
done

echo "===== ALL DONE $(date) ====="
