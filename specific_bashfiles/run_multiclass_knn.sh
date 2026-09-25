
set -euo pipefail
source /.../TensionTRAC/run_env.sh

cd "${REPO}"
mkdir -p results results/figures

echo "=== Multiclass KNN (levels 1-4): video 4-fold (configs/knn_video_split_multiclass.yaml) ==="
"${PYTHON}" src/knn_evaluation/knn.py \
  --config configs/knn_video_split_multiclass.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Multiclass KNN (levels 1-4): stratified 5-fold (configs/knn_stratified_split_multiclass.yaml) ==="
"${PYTHON}" src/knn_evaluation/knn.py \
  --config configs/knn_stratified_split_multiclass.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Figures and k=3 summary ==="
"${PYTHON}" src/knn_evaluation/plot_k_sensitivity.py --task multiclass
"${PYTHON}" src/knn_evaluation/print_metrics_table.py \
  --classifier knn \
  --task multiclass \
  --k 3 \
  --output results/metrics_summary_multiclass_knn_k3.txt

echo ""
echo "Done."
ls -lh results/knn_video_split_multiclass.csv results/knn_stratified_split_multiclass.csv
ls -lh results/metrics_summary_multiclass_knn_k3.txt
