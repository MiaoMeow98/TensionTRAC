set -euo pipefail

REPO=/.../TensionTRAC
BASH_DIR="${REPO}/specific_bashfiles"
source "${REPO}/run_env.sh"

EMB_DIR="${TRAC_DEFAULT_EMBED}"
if [[ ! -d "${EMB_DIR}" ]] || [[ -z "$(ls -A "${EMB_DIR}" 2>/dev/null || true)" ]]; then
  echo "TensionTRAC embeddings missing at ${EMB_DIR}; run: sbatch ${REPO}/run_extract_tension_trac.sh"
  exit 1
fi

bash "${BASH_DIR}/run_binary_mlp.sh"
bash "${BASH_DIR}/run_multiclass_mlp.sh"
bash "${BASH_DIR}/run_cascade_mlp.sh"

cd "${REPO}"
"${PYTHON}" src/knn_evaluation/print_metrics_table.py \
  --classifier mlp \
  --task all \
  --seed 0 \
  --output results/metrics_summary_all_mlp.txt

