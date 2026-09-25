
set -euo pipefail
source /.../TensionTRAC/run_env.sh

cd "${REPO}"
mkdir -p results results/figures

echo "=== Cascade KNN (levels 0-4): video 4-fold (configs/knn_cascade_video_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/knn_cascade.py \
  --config configs/knn_cascade_video_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Cascade KNN (levels 0-4): stratified 5-fold (configs/knn_cascade_stratified_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/knn_cascade.py \
  --config configs/knn_cascade_stratified_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Figures and k=3 summary (E2E + Step1 + Step2) ==="
"${PYTHON}" src/knn_evaluation/plot_k_sensitivity.py --task cascade
"${PYTHON}" src/knn_evaluation/plot_k_sensitivity.py --task cascade_step1
"${PYTHON}" src/knn_evaluation/plot_k_sensitivity.py --task cascade_step2
"${PYTHON}" src/knn_evaluation/print_metrics_table.py \
  --classifier knn \
  --task cascade_all \
  --k 3 \
  --output results/metrics_summary_cascade_knn_k3.txt

echo ""
echo "Done."
ls -lh results/knn_cascade_video_split.csv results/knn_cascade_stratified_split.csv
ls -lh results/metrics_summary_cascade_knn_k3.txt
