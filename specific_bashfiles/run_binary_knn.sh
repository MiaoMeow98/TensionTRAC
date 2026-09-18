set -euo pipefail
source /scratch/p314485/tension_recognition_in_robot_assisted_surgery_SSS/run_env.sh

cd "${REPO}"
mkdir -p results results/figures

echo "=== Binary KNN: video 4-fold (configs/knn_video_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/knn.py \
  --config configs/knn_video_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Binary KNN: stratified 5-fold (configs/knn_stratified_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/knn.py \
  --config configs/knn_stratified_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Figures and k=3 summary ==="
"${PYTHON}" src/knn_evaluation/plot_k_sensitivity.py --task binary
"${PYTHON}" src/knn_evaluation/print_metrics_table.py \
  --classifier knn \
  --task binary \
  --k 3 \
  --output results/metrics_summary_binary_knn_k3.txt

echo ""
echo "Done."
ls -lh results/knn_video_split.csv results/knn_stratified_split.csv
ls -lh results/metrics_summary_binary_knn_k3.txt
