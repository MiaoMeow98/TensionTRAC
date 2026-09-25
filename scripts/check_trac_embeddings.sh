#!/bin/bash
# Verify TensionTRAC embedding directories exist and sample dim matches fusion width.
# Usage:
#   bash scripts/check_trac_embeddings.sh              # default variant only
#   bash scripts/check_trac_embeddings.sh all        # all 7 variants
#   bash scripts/check_trac_embeddings.sh motion     # cross/intra/intra_cross/intra_cross_sem
#   bash scripts/check_trac_embeddings.sh 2d_intra_cross

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO}/run_env.sh"

MODE="${1:-default}"

if [[ "${MODE}" == "all" ]]; then
  VARIANTS=(2d_sem 2d_intra 2d_cross 2d_intra_sem 2d_cross_sem 2d_intra_cross 2d_intra_cross_sem)
elif [[ "${MODE}" == "motion" ]]; then
  VARIANTS=(2d_cross 2d_intra 2d_intra_cross 2d_intra_cross_sem)
elif [[ "${MODE}" == "default" ]]; then
  VARIANTS=("${TRAC_DEFAULT_VARIANT}")
else
  VARIANTS=("$@")
fi

"${PYTHON}" - "${TRAC_FEATURES_ROOT}" "${MODE}" "${VARIANTS[@]}" <<'PY'
import sys
from pathlib import Path

import torch

from src.trajectory_based_model.checkpoint_variants import expected_fusion_dim

features_root = Path(sys.argv[1])
variants = sys.argv[3:]

for variant in variants:
    root = features_root / variant
    files = sorted(root.glob("*/embedding.pt"))
    if not files:
        raise SystemExit(f"Missing embeddings under {root}")

    expected = expected_fusion_dim(variant)
    sample = torch.load(files[0], map_location="cpu", weights_only=False)
    actual = int(torch.as_tensor(sample).reshape(-1).numel())
    if actual != expected:
        raise SystemExit(
            f"[{variant}] expected dim={expected}, got {actual} in {files[0]}"
        )
    print(f"[{variant}] OK: {len(files)} embeddings, dim={actual}")
PY

echo "TensionTRAC embedding check passed (${MODE})."
