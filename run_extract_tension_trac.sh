set -euo pipefail
source /.../TensionTRAC/run_env.sh

cd "${REPO}"
mkdir -p results

VARIANT="${TRAC_VARIANT:-${TRAC_DEFAULT_VARIANT}}"
OVERWRITE="${TRAC_OVERWRITE:-0}"
EXTRACT_ARGS=(--data-root "${DATA_ROOT}" --trajectory-root "${TRAJECTORY_ROOT}" --variant "${VARIANT}")
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRACT_ARGS+=(--overwrite)
fi

echo "=== Extract TensionTRAC embeddings (variant=${VARIANT}, overwrite=${OVERWRITE}) ==="
echo "Trajectory: ${TRAJECTORY_ROOT}"
echo "Annotations: ${DATA_ROOT}/annotations/tension_clip_annotations total 3824.csv"
echo "Output dir: ${DATA_ROOT}/data/TRAC_features/${VARIANT}/"
echo "Feature: ST patch tokens + visibility-weighted mean pool -> [D] (fusion dim)"
"${PYTHON}" src/feature_extraction/tension_trac_tension_inference.py "${EXTRACT_ARGS[@]}"

if [[ "${VARIANT}" == "all" ]]; then
  bash scripts/check_trac_embeddings.sh all
else
  bash scripts/check_trac_embeddings.sh "${VARIANT}"
fi

echo "Done. Embeddings: ${DATA_ROOT}/data/TRAC_features/${VARIANT}/"
