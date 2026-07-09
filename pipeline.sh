export CUDA_VISIBLE_DEVICES=0

# Dataset toggle: `DATASET=ett` or `DATASET=ili` or `DATASET=weather`.
DATASET=${DATASET:-ett}
CONFIG_PATH=${CONFIG_PATH:-}
PRED_LENS=${PRED_LENS:-}

# Batch-size defaults (can override when launching: e.g. BACKBONE_BS=128 sh pipeline.sh)
if [ "${DATASET}" = "ili" ]; then
  BACKBONE_BS=${BACKBONE_BS:-32}      # TCN/Autoformer training batch size
  EVAL_BS=${EVAL_BS:-32}             # eval_degradation.py DataLoader batch size
  HEAD_BS=${HEAD_BS:-32}             # offline Bayesian head training batch size (eval_degradation_ompb.py)
  SRC_BS=${SRC_BS:-32}               # OMPB online source minibatch size (ompb/online_calibration.py)
elif [ "${DATASET}" = "weather" ]; then
  BACKBONE_BS=${BACKBONE_BS:-256}
  EVAL_BS=${EVAL_BS:-256}
  HEAD_BS=${HEAD_BS:-256}
  SRC_BS=${SRC_BS:-256}
else
  BACKBONE_BS=${BACKBONE_BS:-256}
  EVAL_BS=${EVAL_BS:-256}
  HEAD_BS=${HEAD_BS:-256}
  SRC_BS=${SRC_BS:-256}
fi
PROGRESS=${PROGRESS:-1}             # 1 shows tqdm, 0 disables
RETRAIN=${RETRAIN:-0}               # 0 reuses saved checkpoints, 1 retrains from scratch

if [ "${DATASET}" = "ili" ]; then
  CONFIG_PATH=${CONFIG_PATH:-configs/ili.yaml}
  PRED_LENS=${PRED_LENS:-"24 48 72"}
elif [ "${DATASET}" = "weather" ]; then
  CONFIG_PATH=${CONFIG_PATH:-configs/weather.yaml}
  PRED_LENS=${PRED_LENS:-"24 48 96 192 336 720"}
else 
  PRED_LENS=${PRED_LENS:-"24 48 96 192 336 720"}
fi

EXTRA_ARGS=""
if [ -n "${CONFIG_PATH}" ]; then
  EXTRA_ARGS="config_path=${CONFIG_PATH}"
fi

for pred_len in ${PRED_LENS}; do
  echo "=== Running dataset=${DATASET} pred_len=${pred_len} ==="
  python scripts/eval_degradation.py dataset="${DATASET}" model=all models=gpt4ts \
    backbone_batch_size="${BACKBONE_BS}" eval_batch_size="${EVAL_BS}" \
    ${EXTRA_ARGS} pred_len="${pred_len}" retrain="${RETRAIN}"

  python scripts/eval_degradation_ompb.py dataset="${DATASET}" model=all models=tcn,autoformer,gpt4ts\
    backbone_batch_size="${BACKBONE_BS}" head_batch_size="${HEAD_BS}" src_batch_size="${SRC_BS}" \
    ${EXTRA_ARGS} pred_len="${pred_len}" progress="${PROGRESS}" retrain="${RETRAIN}"
done
