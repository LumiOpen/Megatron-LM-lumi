#!/bin/bash

#SBATCH --job-name=v3-debug-val-64-70B
#SBATCH --nodes=64
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=00-04:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000319
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p workdir
wd=$(realpath workdir)

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
mv logs/${SLURM_JOB_ID}.out logs/${SLURM_JOB_ID}-${timestamp}.out
mv logs/${SLURM_JOB_ID}.err logs/${SLURM_JOB_ID}-${timestamp}.err
# symlink logs/latest.out and logs/latest.err
ln -f -s "${SLURM_JOB_ID}-${timestamp}.out" logs/latest.out
ln -f -s "${SLURM_JOB_ID}-${timestamp}.err" logs/latest.err

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup

CONTAINER="/scratch/project_462000319/containers/flashattention_v2_new"
#CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif
#CONTAINER="/flash/project_462000424/singularity/container_out3.sif"
SING_BIND="/scratch/project_462000319,/flash/project_462000319,/scratch/project_462000086,/scratch/project_462000444,/scratch/project_462000353"

LEARNING_RATE=3.2e-4

set -euo pipefail

CHECKPOINT_PATH=checkpoints
TENSORBOARD_PATH="tensorboard/v3-train.$SLURM_JOB_ID"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# sets TRAIN_DATA and VALIDATION_DATA
source europa_data_flash.sh

TRAIN_DATA=$VALIDATION_DATA

MERGES=/flash/project_462000319/europa-tokenizer/merges.txt
VOCAB=/flash/project_462000319/europa-tokenizer/vocab.json

NLAYERS=80
NHIDDEN=8192
NHEADS=64
FFN_HIDDEN_SIZE=28672
SEQ_LEN=5120

MICRO_BATCH_SIZE=1

# Global batch size in tokens ~5.25M
GLOBAL_BATCH_SIZE=$((SLURM_JOB_NUM_NODES * 2))

PP_SIZE=8
TP_SIZE=8
VPP_SIZE=2

# export MEMORY_OPT_ALLREDUCE_SIZE=150000000
# echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

TOTAL_TOKENS=3_000_000_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
# LR_WARMUP_SAMPLES=2000
LR_WARMUP_SAMPLES=$((2000*GLOBAL_BATCH_SIZE))
# I'm setting this to:
NUM_QUERY_GROUPS=8

LOG_INTERVAL=1
SAVE_INTERVAL=1000
EVAL_INTERVAL=4000
EVAL_STEPS=100

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --lr $LEARNING_RATE \
    --min-lr 3e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --use-distributed-optimizer"


GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --bf16 \
    --disable-bias-linear \
    --init-method-std 0.0048 \
    --make-vocab-size-divisible-by 128 \
    --no-gradient-accumulation-fusion \
    --normalization RMSNorm \
    --seed 42 \
    --untie-embeddings-and-output-weights \
    --use-flash-attn \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-query-key-layer-scaling \
    --use-rotary-position-embeddings \
    --no-bias-dropout-fusion \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    $OPTIMIZER_ARGS \
    "
#    --no-async-tensor-model-parallel-allreduce \



OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "
PARALLEL_ARGS="\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
"

if (( VPP_SIZE > 1)); then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP_SIZE"
fi


CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $OUTPUT_ARGS \
    --train-data-path $TRAIN_DATA \
    --valid-data-path $VALIDATION_DATA \
    --dataloader-type single \
    --num-workers 1 \
    --recompute-activations \
    --distributed-timeout-minutes 60 \
    "

echo $CMD


c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
#BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

# add a pythonuserbase to an empty dir to avoid problems with user's local
# python install being imported into the singularity container.
mkdir -p pythonuserbase
export PYTHONUSERBASE=pythonuserbase

echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    -B "$SING_BIND" \
    -B "$PWD" \
    "$CONTAINER" \
    ./launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
