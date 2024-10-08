#!/usr/bin/env bash
#--pre-batch 2 \
#--use-self-negative \
set -x
set -e

TASK="wiki5m_ind"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model '../Model/bert-base-uncased' \
--pooling mean \
--lr 3e-5 \
--seed 42 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task "${TASK}" \
--batch-size 10 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--finetune-t \
--pre-batch 0 \
--epochs 1 \
--workers 3 \
--max-to-keep 2 "$@"
