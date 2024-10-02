#!/bin/bash

#SBATCH --job-name=europa_33B
#SBATCH --nodes=128
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=02-00:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=logs-33B/%j.out
#SBATCH --error=logs-33B/%j.err
#SBATCH --exclude=nid005002,nid005147,nid005150,nid005179,nid005180,nid005348,nid005383,nid005384,nid005513,nid005743,nid005868,nid006282,nid007058,nid007151,nid007249,nid007727,nid005574,nid006706,nid005797

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

mkdir -p logs-33B

mkdir -p workdir
wd=$(realpath workdir)

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
mv logs-33B/${SLURM_JOB_ID}.out logs-33B/${SLURM_JOB_ID}-${timestamp}.out
mv logs-33B/${SLURM_JOB_ID}.err logs-33B/${SLURM_JOB_ID}-${timestamp}.err

# Check if this is a restared run and if so, print the failure events/reasons for failed nodes
if [[ -v SLURM_RESTART_COUNT ]]
then
	failed_node=$(grep 'Node failure' logs-33B/latest.err | awk '{print $NF}')
	if [[ -z ${failed_node:+x} ]]
	then
		echo "RUN RESTARTED but no node failure logged"
	else
		failed_node="${failed_node//$'\n'/ }"
		echo "RUN RESTARTED AFTER FAILURE OF NODE(s) $failed_node. Reason:"
		sacctmgr show event where node="$failed_node" format="NodeName,TimeStart,TimeEnd,State,Reason%100"
	fi
fi

# symlink logs/latest.out and logs/latest.err
ln -f -s "${SLURM_JOB_ID}-${timestamp}.out" logs-33B/latest.out
ln -f -s "${SLURM_JOB_ID}-${timestamp}.err" logs-33B/latest.err

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup
# Container with TCPStore patch.
CONTAINER="/scratch/project_462000353/containers/flashattention_v2_new"
SING_BIND="/scratch/project_462000353/,/flash/project_462000319/"

# LR from LLaMa 2 70B.
LEARNING_RATE=1.5e-4
MIN_LR=1.5e-5

set -euo pipefail

CHECKPOINT_PATH=/scratch/project_462000353/europa-checkpoints/33B_checkpoints
TENSORBOARD_PATH="/scratch/project_462000353/europa-production/tensorboard/v3-33B.$SLURM_JOB_ID"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# sets TRAIN_DATA and VALIDATION_DATA
source europa_data.sh
MERGES=/scratch/project_462000353/europa-tokenizer/merges.txt
VOCAB=/scratch/project_462000353/europa-tokenizer/vocab.json


PP_SIZE=4
TP_SIZE=4
VPP_SIZE=2


MICRO_BATCH_SIZE=1

GLOBAL_BATCH_SIZE=1024
NLAYERS=56
NHIDDEN=7168
NHEADS=56
FFN_HIDDEN_SIZE=20480
SEQ_LEN=4096


TOTAL_TOKENS=3_000_000_000_000
# TOTAL_TOKENS=3_000_000_000
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
    --no-gradient-accumulation-fusion \
    --normalization RMSNorm \
    --seed 42 \
    --untie-embeddings-and-output-weights \
    --use-flash-attn \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-query-key-layer-scaling \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-rotary-position-embeddings \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --overlap-p2p-communication \
    --recompute-activations \
    $OPTIMIZER_ARGS \
    "
    # --overlap-grad-reduce \
    # --standalone-embedding-stage \
    # TODO this was a non-standard option after LUMI came back up, not sure why
    # it was used.
    #--data-cache-path data_cache/run1 \

OUTPUT_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
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
    --num-workers 0 \
    --distributed-timeout-minutes 180 \
    "

echo $CMD


# XXX TODO remove this after testing, maybe?
#HIP_LAUNCH_BLOCKING=1

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
