#!/bin/bash
#SBATCH --job-name=test-europa_33B
#SBATCH --nodes=8
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000319
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


#set -ex pipefail
mkdir -p logs
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
#ln -s "${SLURM_JOB_ID}-${timestamp}.out" logs/latest.out
#ln -s "${SLURM_JOB_ID}-${timestamp}.err" logs/latest.err

export EBU_USER_PREFIX="/projappl/project_462000353/Easybuild"
module purge
module load LUMI/24.03 partition/G
module load PyTorch/2.3.1-rocm-6.0.3-python-3.12-singularity-20240926
module load PrgEnv-amd #Clang compiler for the data indexing

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

wd=(`pwd`)

# LR from LLaMa 2 70B.
LEARNING_RATE=1.5e-4
MIN_LR=1.5e-5

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=/scratch/project_462000353/villekom/europa-checkpoints/33B_checkpoints
TENSORBOARD_PATH="/scratch/project_462000353/villekom/logs/europa-production/tensorboard/meg-ds-33B.$SLURM_JOB_ID"


# sets TRAIN_DATA and VALIDATION_DATA
source $wd/europa_data.sh
MERGES=/scratch/project_462000353/europa-tokenizer/merges.txt
VOCAB=/scratch/project_462000353/europa-tokenizer/vocab.json


PP_SIZE=4
TP_SIZE=4
VPP_SIZE=2


MICRO_BATCH_SIZE=1

GLOBAL_BATCH_SIZE=64
NLAYERS=24
NHIDDEN=2048
NHEADS=16
FFN_HIDDEN_SIZE=5504
SEQ_LEN=2048


TOTAL_TOKENS=8_000_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES

# 2000 steps from LLaMa 2.
LR_WARMUP_SAMPLES=$((2000*GLOBAL_BATCH_SIZE))
# Also from LLaMa 2.
NUM_QUERY_GROUPS=8

LOG_INTERVAL=1
SAVE_INTERVAL=1000
EVAL_INTERVAL=10000
EVAL_STEPS=100

# These are the same as LLaMa 2.
OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --lr $LEARNING_RATE \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --use-distributed-optimizer \
    "

#export TRAIN_DATA="0.04000383024140028 /scratch/project_462000353/europa-data/europa-final-merge/bg/culturax-hplt/merged-train.jsonl_text_document"
#export VALIDATION_DATA="0.04000383024140028 /scratch/project_462000353/europa-data/europa-final-merge/bg/culturax-hplt/merged-validate.jsonl_text_document"



# sqrt(2/(NHIDDEN*5)), from https://github.com/bigscience-workshop/bigscience/blob/master/train/lessons-learned.md
INIT_METHOD_STD=0.00747017

# These arguments are from LLaMa 2.
# Our code has a hard-coded fix for bf16 where RoPE inverse frequency uses fp32.
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
    --init-method-std $INIT_METHOD_STD \
    --make-vocab-size-divisible-by 128 \
    --normalization RMSNorm \
    --seed 42 \
    --untie-embeddings-and-output-weights \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --no-query-key-layer-scaling
    --use-flash-attn \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-rotary-position-embeddings \
    --overlap-p2p-communication \
    --recompute-activations \
    --no-gradient-accumulation-fusion \
    $OPTIMIZER_ARGS \
    "
OUTPUT_ARGS=" \
    --save $CHECKPOINT_PATH \
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

ZERO_STAGE=0



CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $OUTPUT_ARGS \
    --train-data-path $TRAIN_DATA \
    --valid-data-path $VALIDATION_DATA \
    --dataloader-type single \
    --num-workers 7 \
    "

echo $CMD


#DEBUGGING
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=1
#export NCCL_DEBUG=INFO
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1
#export NCCL_DEBUG_SUBSYS=INIT,COLL
export HIP_LAUNCH_BLOCKING=1

c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"


echo "START $SLURM_JOBID: $(date)"
#echo "Python command given: \n"
#echo "$CMD"

#export cc=$(which cc)
export CC=clang
export CXX=clang++
srun --label \
    singularity exec $SIF \
    conda-python-distributed $CMD


echo "END $SLURM_JOBID: $(date)"
