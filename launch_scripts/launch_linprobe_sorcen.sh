#!/bin/bash

#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=acc_resa
#SBATCH --output=slurm_output/job_%j.out
#SBATCH --error=slurm_output/job_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4

export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

module load singularity

echo "START TIME: $(date)"

export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=43305
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
SIF=/path/to/singularity/infer.sif

# ---- Edit these before submitting ----
FINETUNE=/path/to/sorcen_checkpoint/checkpoint-1599.pth
DATA_PATH=/path/to/imagenet
NB_CLASSES=1000
OUTPUT_DIR=${REPO}/lp_IN1k/
# --------------------------------------

echo $OUTPUT_DIR
mkdir -v $OUTPUT_DIR
cp -v ${REPO}/main_linprobe.py $OUTPUT_DIR
cp -v ${REPO}/launch_scripts/launch_linprobe.sh $OUTPUT_DIR

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID singularity exec --nv -B ${REPO} --pwd ${REPO} ${SIF} \
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
    --finetune $FINETUNE \
    --data_path $DATA_PATH \
    --nb_classes $NB_CLASSES \
    --output_dir $OUTPUT_DIR \
    --epochs 90 \
    --warmup_epochs 10 \
    --blr 0.1 --weight_decay 0.0 \
    --batch_size 128 \
    --global_pool \
    --dist_eval

echo "END TIME: $(date)"
