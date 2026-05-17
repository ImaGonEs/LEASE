#!/bin/bash

#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=acc_debug
#SBATCH --output=slurm_output/job_%j.out
#SBATCH --error=slurm_output/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# =============================================================================
# Unconditional image generation — Sorcen
#
# Usage:
#   cd /path/to/LEASE
#   sbatch launch_scripts/launch_gen_uncond.sh
#
# Edit the variables below before submitting.
# =============================================================================

# --- required: where the checkpoint lives and its filename ---
CHECKPOINT_FOLDER=/path/to/sorcen_checkpoint
CHECKPOINT_NAME=checkpoint-1599.pth

# --- generation hyperparams ---
TEMP=6.0
NUM_ITER=20
BATCH_SIZE=50
NUM_IMAGES=50000
MODEL=inference_sorcen_vit_base_patch16_single

# --- singularity ---
SIF=/path/to/singularity/infer.sif
REPO=/path/to/LEASE

# =============================================================================

export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SINGULARITY_TMPDIR=/dev/shm/

module load singularity

echo "START TIME: $(date)"

EXPERIMENT_NAME=$(basename "${CHECKPOINT_FOLDER}")
CKPT="${CHECKPOINT_FOLDER}/${CHECKPOINT_NAME}"

if [ ! -d "${CHECKPOINT_FOLDER}" ]; then
    echo "ERROR: checkpoint folder not found: ${CHECKPOINT_FOLDER}" >&2
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

GEN_DIR="${REPO}/generation_results/${EXPERIMENT_NAME}"
mkdir -p "${GEN_DIR}"

CKPT_COPY="${GEN_DIR}/${EXPERIMENT_NAME}_${CHECKPOINT_NAME}"
if [ ! -f "${CKPT_COPY}" ]; then
    echo "Copying checkpoint to: ${CKPT_COPY}"
    cp "${CKPT}" "${CKPT_COPY}"
else
    echo "Checkpoint copy already exists: ${CKPT_COPY}"
fi

CHECKSUM_MD="${GEN_DIR}/checkpoint_provenance.md"
if [ ! -f "${CHECKSUM_MD}" ]; then
    echo "Running checksum..."
    HASH_ORIG=$(md5sum "${CKPT}"      | awk '{print $1}')
    HASH_COPY=$(md5sum "${CKPT_COPY}" | awk '{print $1}')

    if [ "${HASH_ORIG}" != "${HASH_COPY}" ]; then
        echo "ERROR: checksum mismatch — copy may be corrupted!" >&2
        echo "  original: ${HASH_ORIG}  ${CKPT}" >&2
        echo "  copy:     ${HASH_COPY}  ${CKPT_COPY}" >&2
        exit 1
    fi

    cat > "${CHECKSUM_MD}" <<EOF
# Checkpoint Provenance

| | Path | MD5 |
|---|---|---|
| **Original** | \`${CKPT}\` | \`${HASH_ORIG}\` |
| **Copy** | \`${CKPT_COPY}\` | \`${HASH_COPY}\` |

Checksums match. Copy verified on $(date).
EOF
    echo "Provenance recorded: ${CHECKSUM_MD}"
else
    echo "Provenance file already exists: ${CHECKSUM_MD}"
fi

if [ -n "$(ls -A "${GEN_DIR}"/temp* 2>/dev/null)" ]; then
    echo "Skipping generation: output already has content in ${GEN_DIR}"
    exit 0
fi

echo "Checkpoint (original): ${CKPT}"
echo "Checkpoint (copy):     ${CKPT_COPY}"
echo "Experiment:            ${EXPERIMENT_NAME}"
echo "Output dir:            ${GEN_DIR}"

# =============================================================================

srun --wait=60 --kill-on-bad-exit=1 --jobid "${SLURM_JOBID}" \
    singularity exec --nv -B "${REPO}" -B "${CHECKPOINT_FOLDER}" \
    --pwd "${REPO}" "${SIF}" \
    python -u gen_img_unconditional_sorcen.py \
        --ckpt       "${CKPT_COPY}" \
        --model      "${MODEL}" \
        --temp       "${TEMP}" \
        --num_iter   "${NUM_ITER}" \
        --batch_size "${BATCH_SIZE}" \
        --num_images "${NUM_IMAGES}" \
        --output_dir "${GEN_DIR}"

echo "END TIME: $(date)"
