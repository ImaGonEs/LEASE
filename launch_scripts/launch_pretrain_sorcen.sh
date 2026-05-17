#!/bin/bash

#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=acc_resa
#SBATCH --output=slurm_output/job_%j.out
#SBATCH --error=slurm_output/job_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4

export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

module load singularity

echo "START TIME: $(date)"

export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=43304
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

export NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITY_TMPDIR=/dev/shm/

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

echo $SLURMD_NODENAME
echo $MASTER_ADDR
echo $MASTER_PORT
echo $(hostname)

REPO=/path/to/LEASE
SIF=/path/to/singularity/train.sif

OUTPUT_DIR=${REPO}/sorcen_IN1k_1600ep/
echo $OUTPUT_DIR

mkdir -v $OUTPUT_DIR
cp -v ${REPO}/models_lease_tk_1600.py $OUTPUT_DIR
cp -v ${REPO}/main_pretrain_tk_1600.py $OUTPUT_DIR
cp -v ${REPO}/launch_scripts/launch_pretrain_sorcen.sh $OUTPUT_DIR

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID singularity exec --nv -B ${REPO} --pwd ${REPO} ${SIF} \
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
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --method sorcen \
    --output_dir $OUTPUT_DIR \
    --data_path ./token_datasets/tokenized_data_IN1k_VQGAN.pt

echo "END TIME: $(date)"
