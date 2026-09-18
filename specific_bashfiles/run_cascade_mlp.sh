
set -euo pipefail
source /scratch/p314485/tension_recognition_in_robot_assisted_surgery_SSS/run_env.sh

cd "${REPO}"
mkdir -p results

echo "=== Cascade MLP (levels 0-4): video 4-fold (configs/mlp_cascade_video_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/mlp_cascade.py \
  --config configs/mlp_cascade_video_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Cascade MLP (levels 0-4): stratified 5-fold (configs/mlp_cascade_stratified_split.yaml) ==="
"${PYTHON}" src/knn_evaluation/mlp_cascade.py \
  --config configs/mlp_cascade_stratified_split.yaml \
  --root "${DATA_ROOT}"

echo ""
echo "=== Summary (seed=0, E2E + Step1 + Step2) ==="
"${PYTHON}" src/knn_evaluation/print_metrics_table.py \
  --classifier mlp \
  --task cascade_all \
  --seed 0 \
  --output results/metrics_summary_cascade_mlp.txt

echo ""
echo "Done."
ls -lh results/mlp_cascade_video_split.csv results/mlp_cascade_stratified_split.csv
ls -lh results/metrics_summary_cascade_mlp.txt
