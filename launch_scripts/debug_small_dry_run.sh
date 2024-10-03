#!/bin/bash

#SBATCH --job-name=dry-run
#SBATCH --nodes=16
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000319
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

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
export MASTER_PORT=9696
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup

#CONTAINER="/scratch/project_462000319/containers/vaino_flashattention_v2_new"
CONTAINER="/scratch/project_462000319/containers/flashattention_v2_new"
# CONTAINER="/flash/project_462000424/singularity/flashattention_new"
# CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif
# CONTAINER="/flash/project_462000424/singularity/container_out3.sif"
# CONTAINER="/scratch/project_462000319/rluukkon/singularity/flash-attn-test-2_pems_v2.sif"
SING_BIND="/scratch/project_462000319,/flash/project_462000319,/scratch/project_462000086, /scratch/project_462000319, /scratch/project_462000444"

LEARNING_RATE=3.2e-4

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "${SLURM_JOB_ID}.out" logs/latest.out
ln -f -s "${SLURM_JOB_ID}.err" logs/latest.err

CHECKPOINT_PATH=checkpoints
TENSORBOARD_PATH="tensorboard/70B_test.$SLURM_JOB_ID"
#rm -rf "$CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

export CUDA_DEVICE_MAX_CONNECTIONS=1

source europa_data_flash.sh

# TRAIN_DATA="1.0 /scratch/project_462000319/rluukkon/Megatron-DeepSpeed-dev/dataset/parsebank-combined.dedup.filtered.jsonl-with-reg-scores-MT-filtered_text_document"
# TRAIN_DATA="0.5415810341 /flash/project_462000319/megatron-preprocessed-data/train/merged_slimpajama 0.1304808053 /flash/project_462000319/megatron-preprocessed-data/train/merged_finnish 0.004023063515 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.en-fi.jsonl_text_document 0.004016818638 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.fi-en.jsonl_text_document 0.3153543717 /flash/project_462000319/megatron-preprocessed-data/train/starcoder-merged 0.004543906834 /flash/project_462000319/megatron-preprocessed-data/train/train-books_text_document"
#TRAIN_DATA="0.012220654049506422 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/bg 0.00000000893642498948022 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/bs 0.015704810090048035 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/cs 0.005879219445253446 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/da 0.0749165891937554 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/de 0.02385499073085216 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/el 0.07485718390796112 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/es 0.0020968659520771986 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/et 0.008307970809589553 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/fi 0.05930487990031307 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/fr 9.214295873758325e-05 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/ga 8.498065015073429e-06 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/hr 0.012782696858900723 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/hu 0.0005632749185839986 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/is 0.03226913742821717 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/it 0.0035280451932835953 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/lt 0.0018795012112274479 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/lv 7.824566038848572e-05 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/mt 0.020478931092056618 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/nl 0.0035070388972673135 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/no 0.03552524084530773 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/pl 0.04673079230392011 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/pt 0.011512624658058676 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/ro 0.004376074293091748 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sk 0.0019682708195598647 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sl 0.0017946031026425634 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sr 0.01055164159101632 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sv"
MERGES=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/merges.txt
VOCAB=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/vocab.json

NLAYERS=32 #80 
NHIDDEN=4096 #8192
NHEADS=32 #64
FFN_HIDDEN_SIZE=11008 #28672
SEQ_LEN=4096 #5120

MICRO_BATCH_SIZE=1
#GLOBAL_BATCH_SIZE=$((SLURM_JOB_NUM_NODES * 2))
GLOBAL_BATCH_SIZE=64

PP_SIZE=1
TP_SIZE=2
VPP_SIZE=1

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
    --distributed-timeout-minutes 30 \
    $OPTIMIZER_ARGS \
    "
#    --no-async-tensor-model-parallel-allreduce \



    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH \
OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    "
    # --tensorboard-dir $TENSORBOARD_PATH \
    # --tensorboard-queue-size 5 \
    # --log-timers-to-tensorboard \
    # --log-batch-size-to-tensorboard \
    # --log-validation-ppl-to-tensorboard \
    # "
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

# srun \
#     --label \
#     singularity exec \
#     -B /opt/cray:/opt/cray \
#     -B "$wd"/cray-deps:/opt/cray-deps \
#     -B "$wd":/workdir \
#     -B "$SING_BIND" \
#     "$CONTAINER" \
#     bash "cd /scratch/project_462000319/rluukkon/megatron_tests/Megatron-DeepSpeed-jonabur-copy/ &&  launch.sh $CMD"
#
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
