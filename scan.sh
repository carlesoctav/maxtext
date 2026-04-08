#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export MAXTEXT_REPO_ROOT="${REPO_ROOT}"
export HF_HOME="${HF_HOME:-/mnt/carles/.cache}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-src/maxtext/configs/post_train/sft.yml}"

MODEL_NAME="${MODEL_NAME:-gemma3-1b}"
case "${MODEL_NAME}" in
  gemma3-1b)
    DEFAULT_TOKENIZER="google/gemma-3-1b-it"
    DEFAULT_CHECKPOINT_PATH="${REPO_ROOT}/models/gemma31b-scan/0/items"
    ;;
  gemma3-4b) DEFAULT_TOKENIZER="google/gemma-3-4b-it" ;;
  gemma3-12b) DEFAULT_TOKENIZER="google/gemma-3-12b-it" ;;
  gemma3-27b) DEFAULT_TOKENIZER="google/gemma-3-27b-it" ;;
  *)
    echo "Unsupported MODEL_NAME='${MODEL_NAME}'." >&2
    echo "This MaxText checkout only supports gemma3-1b, gemma3-4b, gemma3-12b, and gemma3-27b." >&2
    exit 1
    ;;
esac

DEFAULT_CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH:-}"
LOAD_PARAMETERS_PATH="${LOAD_PARAMETERS_PATH:-${MAXTEXT_CKPT_PATH:-${DEFAULT_CHECKPOINT_PATH}}}"
if [[ -z "${LOAD_PARAMETERS_PATH}" ]]; then
  echo "Set LOAD_PARAMETERS_PATH (or MAXTEXT_CKPT_PATH) to a MaxText checkpoint path." >&2
  exit 1
fi

if [[ "${LOAD_PARAMETERS_PATH}" == *"/unscanned/"* ]]; then
  echo "warning: scan.sh usually expects a scanned checkpoint, but LOAD_PARAMETERS_PATH looks unscanned." >&2
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${DEFAULT_TOKENIZER}}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${REPO_ROOT}/output/${MODEL_NAME}-scan}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-sft-scan-$(date +%Y%m%d-%H%M%S)}"
HF_PATH="${HF_PATH:-carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
HF_EVAL_SPLIT="${HF_EVAL_SPLIT:-train[:1%]}"
TRAIN_DATA_COLUMNS="${TRAIN_DATA_COLUMNS:-['messages']}"
EVAL_DATA_COLUMNS="${EVAL_DATA_COLUMNS:-['messages']}"
HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-${HF_TOKEN:-}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
STEPS="${STEPS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-100}"
ATTENTION="${ATTENTION:-dot_product}"
REMAT_POLICY="${REMAT_POLICY:-none}"

if [[ "${BASE_OUTPUT_DIRECTORY}" != gs://* && "${BASE_OUTPUT_DIRECTORY}" != hf://* ]]; then
  mkdir -p "${BASE_OUTPUT_DIRECTORY}"
fi

# Notes:
# - This uses the standard MaxText trainer rather than Tunix SFT.
# - HF worker_count > 1 is not supported here, so use grain_num_threads rather than num_workers.
# - assistant_loss=False in your Tunix setup maps to sft_train_on_completion_only=False here.
exec "${PYTHON_BIN}" -m maxtext.trainers.pre_train.train "${CONFIG_PATH}" \
  run_name="${RUN_NAME}" \
  base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
  model_name="${MODEL_NAME}" \
  load_parameters_path="${LOAD_PARAMETERS_PATH}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  hf_access_token="${HF_ACCESS_TOKEN}" \
  dataset_type=hf \
  hf_path="${HF_PATH}" \
  train_split="${TRAIN_SPLIT}" \
  hf_eval_split="${HF_EVAL_SPLIT}" \
  train_data_columns="${TRAIN_DATA_COLUMNS}" \
  eval_data_columns="${EVAL_DATA_COLUMNS}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  steps="${STEPS}" \
  learning_rate="${LEARNING_RATE}" \
  max_target_length="${MAX_TARGET_LENGTH}" \
  checkpoint_period="${CHECKPOINT_PERIOD}" \
  eval_interval=-1 \
  scan_layers=true \
  attention="${ATTENTION}" \
  remat_policy="${REMAT_POLICY}" \
  use_tunix_gradient_accumulation=false \
  sft_train_on_completion_only=false \
  packing=true \
  grain_num_threads=8 \
  grain_num_threads_eval=8 \
  "$@"
