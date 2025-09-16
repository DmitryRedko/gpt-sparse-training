#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="offline"

CONFIG="config/train_wikitext.py"
LOGFILE="eval_results.log"

# очистить предыдущий лог
: > "${LOGFILE}"

# --- списки значений ---
RATIOS=(0.2 0.5 0.8 0.9)
TYPES=("masked-activations-layer" "masked-weights-layer")

run_once () {
  local stype="$1"
  local sratio="$2"
  local run_name="${stype}-${sratio}"
  local out_dir="out-${stype}-${sratio}"

  echo "=========================="
  echo "Run: ${run_name}"
  echo "Out dir: ${out_dir}"
  echo "=========================="

  # TRAIN
  python train.py "${CONFIG}" \
    --out_dir="${out_dir}" \
    --sparsity_ratio="${sratio}" \
    --sparsity_type="${stype}" \
    --wandb_run_name="${run_name}"

  # EVAL (resume)
  tmpfile="$(mktemp)"
  python train.py "${CONFIG}" \
    --eval_only=True \
    --init_from='resume' \
    --out_dir="${out_dir}" \
    --sparsity_ratio="${sratio}" \
    --sparsity_type="${stype}" \
    --wandb_run_name="${run_name}" | tee "${tmpfile}"

  # --- парсинг результатов -------------------------------------------------
  # 1) step / train / val losses
  step_line="$(grep -E 'step[[:space:]]+[0-9]+:' "${tmpfile}" | tail -n1 || true)"
  if [[ -n "${step_line}" && "${step_line}" =~ step[[:space:]]+([0-9]+):[[:space:]]+train[[:space:]]+loss[[:space:]]+([0-9.]+),[[:space:]]+val[[:space:]]+loss[[:space:]]+([0-9.]+) ]]; then
    step_val="${BASH_REMATCH[1]}"
    train_loss="${BASH_REMATCH[2]}"
    val_loss="${BASH_REMATCH[3]}"
  else
    step_val="N/A"
    train_loss="N/A"
    val_loss="N/A"
  fi

  # 2) perplexity
  ppl_line="$(grep -E 'Strict perplexity over full val\.bin:' "${tmpfile}" | tail -n1 || true)"
  if [[ -n "${ppl_line}" && "${ppl_line}" =~ Strict\ perplexity\ over\ full\ val\.bin:\ ([0-9.]+) ]]; then
    ppl="${BASH_REMATCH[1]}"
  else
    ppl="N/A"
  fi

  rm -f "${tmpfile}"

  # --- записываем в общий лог ----------------------------------------------
  {
    echo "${run_name}:"
    printf '\tstep: %s\n'        "${step_val}"
    printf '\ttrain loss: %s\n'  "${train_loss}"
    printf '\tval loss: %s\n'    "${val_loss}"
    printf '\tppl: %s\n'         "${ppl}"
    echo
  } | tee -a "${LOGFILE}"
}

# базовый запуск


# сетка
for stype in "${TYPES[@]}"; do
  for sratio in "${RATIOS[@]}"; do
    run_once "${stype}" "${sratio}"
  done
done

run_once "orig" "0.0"

echo "Все эксперименты завершены. Итоги см. в ${LOGFILE}"
