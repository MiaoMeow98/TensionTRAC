
set -euo pipefail
source /scratch/p314485/tension_recognition_in_robot_assisted_surgery_SSS/run_env.sh

cd "${REPO}"
mkdir -p results

echo "=== Multiclass MLP (levels 1-4): video 4-fold (configs/mlp_video_split_multiclass.yaml) ==="
"${PYTHON}" src/knn_evaluation/evaluate.py \
  --classifier mlp \
  --config configs/mlp_video_split_multiclass.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Multiclass MLP (levels 1-4): stratified 5-fold (configs/mlp_stratified_split_multiclass.yaml) ==="
"${PYTHON}" src/knn_evaluation/evaluate.py \
  --classifier mlp \
  --config configs/mlp_stratified_split_multiclass.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Summary (seed=0) ==="
"${PYTHON}" src/knn_evaluation/print_metrics_table.py \
  --classifier mlp \
  --task multiclass \
  --seed 0 \
  --output results/metrics_summary_multiclass_mlp.txt

echo ""
echo "Done."
ls -lh results/mlp_video_split_multiclass.csv results/mlp_stratified_split_multiclass.csv
ls -lh results/metrics_summary_multiclass_mlp.txt
