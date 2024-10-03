#!/bin/bash

#SBATCH --job-name=v3-train
#SBATCH --nodes=512
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=02-00:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000319
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclude=nid005003,nid007971,nid007972

mkdir -p workdir
wd=$(realpath workdir)


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
SING_BIND="/scratch/project_462000319,/flash/project_462000319,/scratch/project_462000086,/scratch/project_462000444"

LEARNING_RATE=3.2e-4

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "${SLURM_JOB_ID}.out" logs/latest.out
ln -f -s "${SLURM_JOB_ID}.err" logs/latest.err

CHECKPOINT_PATH=checkpoints
TENSORBOARD_PATH="tensorboard/v3-train.$SLURM_JOB_ID"
#rm -rf "$CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

export CUDA_DEVICE_MAX_CONNECTIONS=1

TRAIN_DATA='0.039968583288506304 /scratch/project_462000086/europa-final-merge/bg/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/cs/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/da/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/de/redpajama-v2/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/el/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/en/redpajama-v2/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/es/redpajama-v2/merged-train.jsonl_text_document, 0.007745701149276122 /scratch/project_462000086/europa-final-merge/et/hplt-contrib/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/fi/crawls/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/fr/redpajama-v2/merged-train.jsonl_text_document, 0.0005447996496571791 /scratch/project_462000086/europa-final-merge/ga/culturax-hplt/merged-train.jsonl_text_document, 0.013904632639284669 /scratch/project_462000086/europa-final-merge/hr/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/hu/culturax-hplt/merged-train.jsonl_text_document, 0.0033412003510697296 /scratch/project_462000086/europa-final-merge/is/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/it/redpajama-v2/merged-train.jsonl_text_document, 0.020412857629656434 /scratch/project_462000086/europa-final-merge/lt/culturax-hplt/merged-train.jsonl_text_document, 0.01129090532616734 /scratch/project_462000086/europa-final-merge/lv/culturax-hplt/merged-train.jsonl_text_document, 0.00040381714286352396 /scratch/project_462000086/europa-final-merge/mt/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/nl/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/no/hplt-contrib/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/pl/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/pt/culturax-hplt/merged-train.jsonl_text_document, 0.03996858328850631 /scratch/project_462000086/europa-final-merge/ro/culturax-hplt/merged-train.jsonl_text_document, 0.025735578998216015 /scratch/project_462000086/europa-final-merge/sk/culturax-hplt/merged-train.jsonl_text_document, 0.012614411433174915 /scratch/project_462000086/europa-final-merge/sl/culturax-hplt/merged-train.jsonl_text_document, 0.039968583288506304 /scratch/project_462000086/europa-final-merge/sv/culturax-hplt/merged-train.jsonl_text_document, 0.000953432927202873 /scratch/project_462000086/europa-final-merge/multiling/europarl/merged-train.jsonl_text_document, 0.01725756619131265 /scratch/project_462000086/europa-final-merge/multiling/wikipedia/merged-train.jsonl_text_document, 0.0010896906608343251 /scratch/project_462000086/europa-final-merge/fi/news-lonnrot/merged-train.jsonl_text_document, 0.001602259071045411 /scratch/project_462000086/europa-final-merge/fi/suomi24/merged-train.jsonl_text_document, 0.0692788777000776 /scratch/project_462000086/europa-final-merge/code/starcoder/merged-validate.jsonl_text_document/code/starcoder/merged-train.jsonl_text_document, 0.000733268507176527 /scratch/project_462000086/europa-final-merge/en/british-library/merged-validate.jsonl_text_document/en/british-library/merged-train.jsonl_text_document, 0.0075201113487895375 /scratch/project_462000086/europa-final-merge/en/cosmopedia/merged-validate.jsonl_text_document/en/cosmopedia/merged-train.jsonl_text_document, 0.00025002007609781473 /scratch/project_462000086/europa-final-merge/en/natural_instructions/merged-validate.jsonl_text_document/en/natural_instructions/merged-train.jsonl_text_document, 0.020711022008306677 /scratch/project_462000086/europa-final-merge/en/peS2o/merged-validate.jsonl_text_document/en/peS2o/merged-train.jsonl_text_document, 0.02417217305291462 /scratch/project_462000086/europa-final-merge/en/pile-of-law/merged-validate.jsonl_text_document/en/pile-of-law/merged-train.jsonl_text_document, 0.0015753216148329354 /scratch/project_462000086/europa-final-merge/en/pubmed/merged-validate.jsonl_text_document/en/pubmed/merged-train.jsonl_text_document, 0.008259261637678874 /scratch/project_462000086/europa-final-merge/en/pubmed-central/merged-validate.jsonl_text_document/en/pubmed-central/merged-train.jsonl_text_document, 0.04499131396429607 /scratch/project_462000086/europa-final-merge/xling/tatoeba/merged-validate.jsonl_text_document/xling/tatoeba/merged-train.jsonl_text_document, 8.409971500095727e-05 /scratch/project_462000086/europa-final-merge/fr/british-library/merged-validate.jsonl_text_document/fr/british-library/merged-train.jsonl_text_document, 0.026061761300460055 /scratch/project_462000086/europa-final-merge/en/reddit/merged-validate.jsonl_text_document/en/reddit/merged-train.jsonl_text_document'
MERGES=/scratch/project_462000444/europa/tokenizers/europa_tokenizer_262144_rc3-sampled-50B-shuf.jsonl/merges.txt
VOCAB=/scratch/project_462000444/europa/tokenizers/europa_tokenizer_262144_rc3-sampled-50B-shuf.jsonl/vocab.json

NLAYERS=80
NHIDDEN=8192
NHEADS=64
FFN_HIDDEN_SIZE=28672
SEQ_LEN=5120

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=4
#GLOBAL_BATCH_SIZE=$((SLURM_JOB_NUM_NODES * 2))

PP_SIZE=8
TP_SIZE=8
VPP_SIZE=2

# export MEMORY_OPT_ALLREDUCE_SIZE=150000000
# echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

TOTAL_TOKENS=3_000_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES/100))

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
    "

DEEPSPEED=false
if [ $DEEPSPEED != true ];then
   OPTIMIZER_ARGS="$OPTIMIZER_ARGS \
   --use-distributed-optimizer"
fi

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

ZERO_STAGE=1

mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/$SLURM_JOB_ID.json"

cat <<EOF > $DS_CONFIG_PATH
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "gradient_clipping": 1.0,
    "bf16": {
        "enabled": true
    },
    "data_types": {
        "grad_accum_dtype": "float32"
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": true
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_CONFIG_PATH \
    --zero-stage $ZERO_STAGE \
    "

DEEPSPEED_ARGS="--deepspeed-activation-checkpointing ${DEEPSPEED_ARGS}"
DEEPSPEED_ARGS="--recompute-granularity selective  ${DEEPSPEED_ARGS}"


CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $OUTPUT_ARGS \
    --data-path $TRAIN_DATA \
    --dataloader-type single \
    --num-workers 1 \
    --recompute-activations \
    "
    # --profile \
    # --profile-step-end 20 \
    
if [ $DEEPSPEED = true ]; then
    CMD="$CMD \
    $DEEPSPEED_ARGS"
fi
    # --valid-data-path $VALIDATION_DATA \

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
