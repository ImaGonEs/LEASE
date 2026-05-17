#!/bin/bash

#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=acc_debug
#SBATCH --output=slurm_output/sanity_%j.out
#SBATCH --error=slurm_output/sanity_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4

# =============================================================================
# Sanity check — all 6 pipelines sequentially in a single job
#
# Tests: LEASE pretrain, Sorcen pretrain,
#        LEASE LP,       Sorcen LP,
#        LEASE gen,      Sorcen gen
#
# Each stage runs 1 epoch (pretrain/LP) or 50 images (gen) — just enough to
# verify the full pipeline runs without errors.
#
# Usage (from server):
#   cd /path/to/LEASE
#   sbatch launch_scripts/launch_sanity_check.sh
# =============================================================================

export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SINGULARITY_TMPDIR=/dev/shm/
export NCCL_ASYNC_ERROR_HANDLING=1

module load singularity

REPO=/path/to/LEASE
SIF_TRAIN=/path/to/singularity/train.sif
SIF_INFER=/path/to/singularity/infer.sif

LEASE_CKPT=/path/to/lease_checkpoint/checkpoint-1599.pth
SORCEN_CKPT=/path/to/sorcen_checkpoint/checkpoint-1599.pth
IMAGENET=/path/to/imagenet

SANITY_DIR=${REPO}/sanity_check_results
mkdir -p ${SANITY_DIR}/{lease_pretrain,sorcen_pretrain,lease_lp,sorcen_lp,lease_gen,sorcen_gen}

export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

SRUN_ARGS="--wait=60 --kill-on-bad-exit=1 --jobid $SLURM_JOBID"

echo "========================================================"
echo "  Sanity check — START: $(date)"
echo "  Nodes: ${NNODES} x ${GPUS_PER_NODE} GPUs"
echo "  REPO: ${REPO}"
echo "========================================================"

# =============================================================================
# 1. LEASE pretrain — 1 epoch
# =============================================================================
echo ""
echo "[1/6] LEASE pretrain (1 epoch) — $(date)"
export MASTER_PORT=43310

srun $SRUN_ARGS singularity exec --nv -B ${REPO} --pwd ${REPO} ${SIF_TRAIN} \
    python -u -m torch.distributed.run \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --max_restarts 0 \
        --role $SLURMD_NODENAME \
        --tee 3 \
        main_pretrain_tk_1600.py \
        --batch_size 64 \
        --tk \
        --model lease_vit_base_patch16_single \
        --mask_ratio_min 0.5 --mask_ratio_max 1.0 \
        --mask_ratio_mu 0.55 --mask_ratio_std 0.25 \
        --epochs 1 \
        --warmup_epochs 0 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --method lease \
        --output_dir ${SANITY_DIR}/lease_pretrain \
        --data_path ./token_datasets/tokenized_data_IN1k_VQGAN_DINOv2.pt

echo "[1/6] LEASE pretrain — DONE: $(date)"

# =============================================================================
# 2. Sorcen pretrain — 1 epoch
# =============================================================================
echo ""
echo "[2/6] Sorcen pretrain (1 epoch) — $(date)"
export MASTER_PORT=43311

srun $SRUN_ARGS singularity exec --nv -B ${REPO} --pwd ${REPO} ${SIF_TRAIN} \
    python -u -m torch.distributed.run \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --max_restarts 0 \
        --role $SLURMD_NODENAME \
        --tee 3 \
        main_pretrain_tk_1600.py \
        --batch_size 128 \
        --tk \
        --model sorcen_vit_base_patch16_single \
        --mask_ratio_min 0.5 --mask_ratio_max 1.0 \
        --mask_ratio_mu 0.55 --mask_ratio_std 0.25 \
        --epochs 1 \
        --warmup_epochs 0 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --method sorcen \
        --output_dir ${SANITY_DIR}/sorcen_pretrain \
        --data_path ./token_datasets/tokenized_data_IN1k_VQGAN.pt

echo "[2/6] Sorcen pretrain — DONE: $(date)"

# =============================================================================
# 3. LEASE LP — 1 epoch
# =============================================================================
echo ""
echo "[3/6] LEASE LP (1 epoch) — $(date)"
export MASTER_PORT=43312

srun $SRUN_ARGS singularity exec --nv \
    -B ${REPO} -B /path/to/checkpoints_root -B ${IMAGENET} \
    --pwd ${REPO} ${SIF_INFER} \
    python -u -m torch.distributed.run \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --max_restarts 0 \
        --role $SLURMD_NODENAME \
        --tee 3 \
        main_linprobe.py \
        --model vit_base_patch16_lease \
        --finetune ${LEASE_CKPT} \
        --data_path ${IMAGENET} \
        --nb_classes 1000 \
        --output_dir ${SANITY_DIR}/lease_lp \
        --epochs 1 \
        --warmup_epochs 0 \
        --blr 0.1 --weight_decay 0.0 \
        --batch_size 128 \
        --global_pool \
        --dist_eval

echo "[3/6] LEASE LP — DONE: $(date)"

# =============================================================================
# 4. Sorcen LP — 1 epoch
# =============================================================================
echo ""
echo "[4/6] Sorcen LP (1 epoch) — $(date)"
export MASTER_PORT=43313

srun $SRUN_ARGS singularity exec --nv \
    -B ${REPO} -B /path/to/checkpoints_root -B ${IMAGENET} \
    --pwd ${REPO} ${SIF_INFER} \
    python -u -m torch.distributed.run \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --max_restarts 0 \
        --role $SLURMD_NODENAME \
        --tee 3 \
        main_linprobe.py \
        --model vit_base_patch16 \
        --finetune ${SORCEN_CKPT} \
        --data_path ${IMAGENET} \
        --nb_classes 1000 \
        --output_dir ${SANITY_DIR}/sorcen_lp \
        --epochs 1 \
        --warmup_epochs 0 \
        --blr 0.1 --weight_decay 0.0 \
        --batch_size 128 \
        --global_pool \
        --dist_eval

echo "[4/6] Sorcen LP — DONE: $(date)"

# =============================================================================
# 5. LEASE generation — 50 images (1 batch)
# =============================================================================
echo ""
echo "[5/6] LEASE generation (50 images) — $(date)"

srun -N1 -n1 --gres=gpu:1 --jobid $SLURM_JOBID \
    singularity exec --nv \
    -B ${REPO} -B /path/to/checkpoints_root \
    --pwd ${REPO} ${SIF_INFER} \
    python -u gen_img_unconditional_lease.py \
        --ckpt       ${LEASE_CKPT} \
        --model      inference_lease_vit_base_patch16_single \
        --temp       6.0 \
        --num_iter   20 \
        --batch_size 50 \
        --num_images 50 \
        --output_dir ${SANITY_DIR}/lease_gen

echo "[5/6] LEASE generation — DONE: $(date)"

# =============================================================================
# 6. Sorcen generation — 50 images (1 batch)
# =============================================================================
echo ""
echo "[6/6] Sorcen generation (50 images) — $(date)"

srun -N1 -n1 --gres=gpu:1 --jobid $SLURM_JOBID \
    singularity exec --nv \
    -B ${REPO} -B /path/to/checkpoints_root \
    --pwd ${REPO} ${SIF_INFER} \
    python -u gen_img_unconditional_sorcen.py \
        --ckpt       ${SORCEN_CKPT} \
        --model      inference_sorcen_vit_base_patch16_single \
        --temp       6.0 \
        --num_iter   20 \
        --batch_size 50 \
        --num_images 50 \
        --output_dir ${SANITY_DIR}/sorcen_gen

echo "[6/6] Sorcen generation — DONE: $(date)"

echo ""
echo "========================================================"
echo "  All 6 stages complete — $(date)"
echo "  Logs → ${SANITY_DIR}"
echo "========================================================"
