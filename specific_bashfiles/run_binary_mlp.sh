

set -euo pipefail
source /.../TensionTRAC/run_env.sh

cd "${REPO}"
mkdir -p results

echo "=== Binary MLP: video 4-fold (configs/mlp_video_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/evaluate.py \
  --classifier mlp \
  --config configs/mlp_video_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Binary MLP: stratified 5-fold (configs/mlp_stratified_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/evaluate.py \
  --classifier mlp \
  --config configs/mlp_stratified_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Summary (seed=0) ==="
"${PYTHON}" src/knn_evaluation/print_metrics_table.py \
  --classifier mlp \
  --task binary \
  --seed 0 \
  --output results/metrics_summary_binary_mlp.txt

echo ""
echo "Done."
ls -lh results/mlp_video_split.csv results/mlp_stratified_split.csv
ls -lh results/metrics_summary_binary_mlp.txt
