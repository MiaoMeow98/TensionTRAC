# Shared paths for all run_*.sh scripts in this repository.
REPO=/.../TensionTRAC
DATA_ROOT=/.../tension_laparoscopic_surgery
PYTHON=/.../virenv/*your_env_name*/bin/python # (env configuration is based on /env_requirements.txt )
TRAC_CHECKPOINT_DIR="${REPO}/TRAC_checkpoints/TensionTRAC_checkpoint"
TRAC_CONFIG="${TRAC_CHECKPOINT_DIR}/TensionTRAC_checkpoint.yaml"
TRAC_CHECKPOINT="${TRAC_CHECKPOINT_DIR}/checkpoint.pyth"
TRAJECTORY_ROOT="${DATA_ROOT}/trajectories/cotracker"
TRAC_FEATURES_ROOT="${DATA_ROOT}/data/TRAC_features"
TRAC_DEFAULT_VARIANT=checkpoint
TRAC_DEFAULT_EMBED="${TRAC_FEATURES_ROOT}/${TRAC_DEFAULT_VARIANT}"
