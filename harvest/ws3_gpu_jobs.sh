#!/usr/bin/env bash
# WS3 sponsor-GPU jobs 1-6 (planned for one L40S). Spends credits: run only after Hasif's go-ahead.
# Every job can be rerun after a stop: finished work is skipped. Logs hold ids, counts and numbers only.
#
#   export HF_TOKEN=...        # set it in the terminal or the provider's secrets; never paste it into a file
#   bash harvest/ws3_gpu_jobs.sh setup
#   bash harvest/ws3_gpu_jobs.sh 1            # timing test: read its speeds before going on
#   bash harvest/ws3_gpu_jobs.sh minimum      # jobs 2, 3, 4
#   bash harvest/ws3_gpu_jobs.sh recommended  # jobs 5, 6
#
# Jobs: 1 timing test | 2 Mistral replies | 3 Gemma replies | 4 WildGuard on both | 5 Mistral C1 lost prompts | 6 Gemma C1
# 7 Qwen2.5-7B bf16, all prompts | 8 Qwen2.5-1.5B bf16, all prompts | bf16 = 7 and 8
# 9 Qwen replies + WildGuard, all prompts (H2 on Qwen) | 10 Gemma and Mistral C1 bf16, all prompts (D1; about 1 GB on Drive)
# Needs on the machine: this repo with data/pairs_table_v1_final-2.csv, and for job 5 the Mistral C1 harvest folder.
# data/ holds harmful text: delete the repo from a rented machine when finished, after copying the outputs back.
#
# A5_WORK        folder that holds the repo (default: the repo's parent, so on Colab MyDrive/a5-work)
# JOB_MAX_HOURS  time limit per job, a guard against spending credits on a stuck run (default 2)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${A5_WORK:-$(dirname "$REPO")}"
PAIRS="$REPO/data/pairs_table_v1_final-2.csv"
HARVEST="$WORK/ws3_harvest"
REPLIES="$REPO/data/ws3_replies"
MISTRAL_C1="$HARVEST/mistral-7b__wj-sorry__hf-bf16-bs1__sub4000"
LOGS="$WORK/logs"
MAX_SECONDS=$(( ${JOB_MAX_HOURS:-2} * 3600 ))
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # the scripts check the GPU with torch before vLLM starts
cd "$REPO"
mkdir -p "$LOGS"

need() {
  [ -n "${HF_TOKEN:-}" ] || { echo "STOP: HF_TOKEN is not set"; exit 1; }
  [ -f "$PAIRS" ] || { echo "STOP: $PAIRS is missing"; exit 1; }
  nvidia-smi -L >/dev/null || { echo "STOP: no GPU"; exit 1; }
}

run() {   # run <log name> <command...>: one job with a time limit and its own log
  local name="$1"; shift
  echo "=== $name: $(date -u +%H:%M:%S) UTC"
  local code=0
  timeout "$MAX_SECONDS" "$@" 2>&1 | tee -a "$LOGS/gpu_$name.log" || code=$?
  if [ "$code" -eq 124 ]; then
    echo "STOP: $name hit the ${JOB_MAX_HOURS:-2} h limit. Saved chunks are kept; rerun this job to resume."
  fi
  [ "$code" -eq 0 ] || exit "$code"
}

job1() {
  run t_mistral python harvest/ws3_generate.py --model mistral-7b --pairs "$PAIRS" --limit 10
  run t_gemma python harvest/ws3_generate.py --model gemma2-9b --pairs "$PAIRS" --limit 10
  run t_judge_mistral python harvest/ws3_judge.py --pairs "$PAIRS" --replies "$REPLIES/mistral-7b__wj-sorry__vllm-bf16__smoke10"
  run t_judge_gemma python harvest/ws3_judge.py --pairs "$PAIRS" --replies "$REPLIES/gemma2-9b__wj-sorry__vllm-bf16__smoke10"
  run t_gemma_c1 python harvest/ws3_harvest.py --model gemma2-9b --config c1 --dtype bfloat16 --limit 5 --sweep-every 42 \
    --pairs "$PAIRS" --out "$WORK/ws3_smoke/gemma2-9b__c1__smoke5"
  if [ -f "$MISTRAL_C1/meta.json" ]; then
    run t_fill python harvest/ws3_fill_lost.py --folder "$MISTRAL_C1" --pairs "$PAIRS" --max-fill 5 --check 5
  else
    echo "job 5's timing test skipped: $MISTRAL_C1 is not on this machine"
  fi
}
job2() { run replies_mistral python harvest/ws3_generate.py --model mistral-7b --pairs "$PAIRS" --subset 4000; }
job3() { run replies_gemma python harvest/ws3_generate.py --model gemma2-9b --pairs "$PAIRS" --subset 4000; }
job4() {
  run judge_mistral python harvest/ws3_judge.py --pairs "$PAIRS" --replies "$REPLIES/mistral-7b__wj-sorry__vllm-bf16__sub4000"
  run judge_gemma python harvest/ws3_judge.py --pairs "$PAIRS" --replies "$REPLIES/gemma2-9b__wj-sorry__vllm-bf16__sub4000"
}
job5() {
  [ -f "$MISTRAL_C1/meta.json" ] || { echo "STOP: $MISTRAL_C1 is not on this machine"; exit 1; }
  run fill_mistral_c1 python harvest/ws3_fill_lost.py --folder "$MISTRAL_C1" --pairs "$PAIRS"
}
job6() {
  run gemma_c1 python harvest/ws3_harvest.py --model gemma2-9b --config c1 --dtype bfloat16 --subset 4000 --sweep-every 42 \
    --pairs "$PAIRS" --out "$HARVEST/gemma2-9b__wj-sorry__hf-bf16-bs1__sub4000"
}
# bf16 runs of the two Qwen models on all prompts (10 Nov), replacing their 4-bit and fp16 runs in the analysis
job7() {
  run qwen7b_c1 python harvest/ws3_harvest.py --model qwen25-7b --config c1 --dtype bfloat16 --sweep-every 4     --pairs "$PAIRS" --out "$HARVEST/qwen25-7b__wj-sorry__hf-bf16-bs1"
}
job8() {
  run qwen1p5b_c1 python harvest/ws3_harvest.py --model qwen25-1p5b --config c1 --dtype bfloat16 --sweep-every 8     --pairs "$PAIRS" --out "$HARVEST/qwen25-1p5b__wj-sorry__hf-bf16-bs1"
}

# 11 Nov: replies and labels for the two Qwen models on all prompts, so H2 can run on their full bf16 folders
job9() {
  run replies_qwen7b python harvest/ws3_generate.py --model qwen25-7b --pairs "$PAIRS"
  run replies_qwen1p5b python harvest/ws3_generate.py --model qwen25-1p5b --pairs "$PAIRS"
  run judge_qwen7b python harvest/ws3_judge.py --pairs "$PAIRS" --replies "$REPLIES/qwen25-7b__wj-sorry__vllm-bf16"
  run judge_qwen1p5b python harvest/ws3_judge.py --pairs "$PAIRS" --replies "$REPLIES/qwen25-1p5b__wj-sorry__vllm-bf16"
}
# 11 Nov: full bf16 harvests of Gemma-2-9B and Mistral-7B (deviation D1); no layer sweep, as on the subset runs
job10() {
  run gemma_c1_full python harvest/ws3_harvest.py --model gemma2-9b --config c1 --dtype bfloat16 --sweep-every 42     --pairs "$PAIRS" --out "$HARVEST/gemma2-9b__wj-sorry__hf-bf16-bs1"
  run mistral_c1_full python harvest/ws3_harvest.py --model mistral-7b --config c1 --dtype bfloat16 --sweep-every 32     --pairs "$PAIRS" --out "$HARVEST/mistral-7b__wj-sorry__hf-bf16-bs1"
}
case "${1:-}" in
  setup)
    python -m pip install -q -r requirements.txt vllm
    nvidia-smi -L
    python -c "import torch, transformers, vllm; print('torch', torch.__version__, '| transformers', transformers.__version__, '| vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"
    ;;
  1) need; job1 ;;
  2) need; job2 ;;
  3) need; job3 ;;
  4) need; job4 ;;
  5) need; job5 ;;
  6) need; job6 ;;
  minimum) need; job2; job3; job4 ;;
  recommended) need; job5; job6 ;;
  7) need; job7 ;;
  8) need; job8 ;;
  bf16) need; job7; job8 ;;
  9) need; job9 ;;
  10) need; job10 ;;
  *) echo "usage: bash harvest/ws3_gpu_jobs.sh setup|1|2|3|4|5|6|7|8|minimum|recommended|bf16"; exit 2 ;;
esac
echo "finished: ${1}. Stop the machine if nothing else is queued, so it stops charging."
