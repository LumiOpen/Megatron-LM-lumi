#!/bin/bash
#SBATCH --job-name=v3-test-start
#SBATCH --nodes=2
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=00-2:00:00
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
# CONTAINER="/flash/project_462000424/singularity/flashattention_new"
#CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif
#CONTAINER="/flash/project_462000424/singularity/container_out3.sif"
# CONTAINER="/scratch/project_462000319/rluukkon/singularity/flash-attn-test-2_pems_v2.sif"
SING_BIND="/scratch/project_462000319,/flash/project_462000319,/scratch/project_462000086,/scratch/project_462000444"

LEARNING_RATE=3.2e-4

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "${SLURM_JOB_ID}.out" logs/latest.out
ln -f -s "${SLURM_JOB_ID}.err" logs/latest.err

CHECKPOINT_PATH=checkpoints
TENSORBOARD_PATH="tensorboard/70B_test.$SLURM_JOB_ID"
#rm -rf "$CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

export CUDA_DEVICE_MAX_CONNECTIONS=1

# TRAIN_DATA="1.0 /scratch/project_462000319/rluukkon/Megatron-DeepSpeed-dev/dataset/parsebank-combined.dedup.filtered.jsonl-with-reg-scores-MT-filtered_text_document"
# TRAIN_DATA="0.5415810341 /flash/project_462000319/megatron-preprocessed-data/train/merged_slimpajama 0.1304808053 /flash/project_462000319/megatron-preprocessed-data/train/merged_finnish 0.004023063515 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.en-fi.jsonl_text_document 0.004016818638 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.fi-en.jsonl_text_document 0.3153543717 /flash/project_462000319/megatron-preprocessed-data/train/starcoder-merged 0.004543906834 /flash/project_462000319/megatron-preprocessed-data/train/train-books_text_document"
TRAIN_DATA='0.028330616902017453 /scratch/project_462000444/europa/tokenized-data/merged-256k/bg/culturax/merged-train.jsonl_text_document, 1.3877952441524676e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/bg/europarl/merged-train.jsonl_text_document, 0.011637966386488853 /scratch/project_462000444/europa/tokenized-data/merged-256k/bg/hplt/merged-train.jsonl_text_document, 0.00018549465667909475 /scratch/project_462000444/europa/tokenized-data/merged-256k/bg/wikipedia/merged-train.jsonl_text_document, 0.06927887770007761 /scratch/project_462000444/europa/tokenized-data/merged-256k/code/starcoder/merged-train.jsonl_text_document, 0.029512568359902513 /scratch/project_462000444/europa/tokenized-data/merged-256k/cs/culturax/merged-train.jsonl_text_document, 2.0198780048827176e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/cs/europarl/merged-train.jsonl_text_document, 0.010456014928603794 /scratch/project_462000444/europa/tokenized-data/merged-256k/cs/hplt/merged-train.jsonl_text_document, 0.00039916076560640197 /scratch/project_462000444/europa/tokenized-data/merged-256k/cs/wikipedia/merged-train.jsonl_text_document, 0.029076253483085727 /scratch/project_462000444/europa/tokenized-data/merged-256k/da/culturax/merged-train.jsonl_text_document, 6.95445755189185e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/da/europarl/merged-train.jsonl_text_document, 0.010892329805420582 /scratch/project_462000444/europa/tokenized-data/merged-256k/da/hplt/merged-train.jsonl_text_document, 0.00013710806723934432 /scratch/project_462000444/europa/tokenized-data/merged-256k/da/wikipedia/merged-train.jsonl_text_document, 7.175635298407953e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/de/europarl/merged-train.jsonl_text_document, 0.03996858328850631 /scratch/project_462000444/europa/tokenized-data/merged-256k/de/redpajama-v2/merged-train.jsonl_text_document, 0.002412177499689165 /scratch/project_462000444/europa/tokenized-data/merged-256k/de/wikipedia/merged-train.jsonl_text_document, 0.021998682222185418 /scratch/project_462000444/europa/tokenized-data/merged-256k/el/culturax/merged-train.jsonl_text_document, 5.953543705360681e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/el/europarl/merged-train.jsonl_text_document, 0.01796990106632089 /scratch/project_462000444/europa/tokenized-data/merged-256k/el/hplt/merged-train.jsonl_text_document, 0.00021355336155070868 /scratch/project_462000444/europa/tokenized-data/merged-256k/el/wikipedia/merged-train.jsonl_text_document, 0.0007332685071765271 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/british-library/merged-train.jsonl_text_document, 0.007520111348789538 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/cosmopedia/merged-train.jsonl_text_document, 6.798419603204772e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/europarl/merged-train.jsonl_text_document, 0.00025002007609781473 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/natural_instructions/merged-train.jsonl_text_document, 0.02071102200830668 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/peS2o/merged-train.jsonl_text_document, 0.024172173052914622 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/pile-of-law/merged-train.jsonl_text_document, 0.0015753216148329358 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/pubmed/merged-train.jsonl_text_document, 0.008259261637678875 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/pubmed-central/merged-train.jsonl_text_document, 0.02606176130046006 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/reddit/merged-train.jsonl_text_document, 0.03996858328850631 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/redpajama-v2/merged-train.jsonl_text_document, 0.004903304282466492 /scratch/project_462000444/europa/tokenized-data/merged-256k/en/wikipedia/merged-train.jsonl_text_document, 7.227729150808693e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/es/europarl/merged-train.jsonl_text_document, 0.03996858328850631 /scratch/project_462000444/europa/tokenized-data/merged-256k/es/redpajama-v2/merged-train.jsonl_text_document, 0.0014543180139169355 /scratch/project_462000444/europa/tokenized-data/merged-256k/es/wikipedia/merged-train.jsonl_text_document, 1.956430976477419e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/et/europarl/merged-train.jsonl_text_document, 0.0077457011492761225 /scratch/project_462000444/europa/tokenized-data/merged-256k/et/hplt-contrib/merged-train.jsonl_text_document, 0.00011477301737369017 /scratch/project_462000444/europa/tokenized-data/merged-256k/et/wikipedia/merged-train.jsonl_text_document, 0.020873354064412392 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/cc-fi/merged-train.jsonl_text_document, 6.543678638557978e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/europarl/merged-train.jsonl_text_document, 0.01271895817724224 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/mc4_fi/merged-train.jsonl_text_document, 0.006376271046851675 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/parsebank/merged-train.jsonl_text_document, 0.00019135621029190195 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/projekti-lonnrot/merged-train.jsonl_text_document, 0.0005312892399807874 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/stt-fi-1992-2018/merged-train.jsonl_text_document, 0.0016022590710454113 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/suomi24/merged-train.jsonl_text_document, 0.00028425907537410196 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/wikipedia/merged-train.jsonl_text_document, 0.000367045210561636 /scratch/project_462000444/europa/tokenized-data/merged-256k/fi/ylenews-fi/merged-train.jsonl_text_document, 8.409971500095728e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/fr/british-library/merged-train.jsonl_text_document, 7.477198559084108e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/fr/europarl/merged-train.jsonl_text_document, 0.03996858328850631 /scratch/project_462000444/europa/tokenized-data/merged-256k/fr/redpajama-v2/merged-train.jsonl_text_document, 0.0019537166000726924 /scratch/project_462000444/europa/tokenized-data/merged-256k/fr/wikipedia/merged-train.jsonl_text_document, 0.00040447953553293393 /scratch/project_462000444/europa/tokenized-data/merged-256k/ga/culturax/merged-train.jsonl_text_document, 0.00014032011412424528 /scratch/project_462000444/europa/tokenized-data/merged-256k/ga/hplt/merged-train.jsonl_text_document, 1.444822718552883e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/ga/wikipedia/merged-train.jsonl_text_document, 5.206583175928301e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/hr/culturax/merged-train.jsonl_text_document, 0.013852566807525389 /scratch/project_462000444/europa/tokenized-data/merged-256k/hr/hplt/merged-train.jsonl_text_document, 0.00011745625825203492 /scratch/project_462000444/europa/tokenized-data/merged-256k/hr/wikipedia/merged-train.jsonl_text_document, 0.028740425987522376 /scratch/project_462000444/europa/tokenized-data/merged-256k/hu/culturax/merged-train.jsonl_text_document, 2.1440797779230018e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/hu/europarl/merged-train.jsonl_text_document, 0.011228157300983933 /scratch/project_462000444/europa/tokenized-data/merged-256k/hu/hplt/merged-train.jsonl_text_document, 0.0003808002366276131 /scratch/project_462000444/europa/tokenized-data/merged-256k/hu/wikipedia/merged-train.jsonl_text_document, 0.0027825343048876695 /scratch/project_462000444/europa/tokenized-data/merged-256k/is/culturax/merged-train.jsonl_text_document, 0.0005586660461820605 /scratch/project_462000444/europa/tokenized-data/merged-256k/is/hplt/merged-train.jsonl_text_document, 2.1449732756024163e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/is/wikipedia/merged-train.jsonl_text_document, 7.043192902786702e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/it/europarl/merged-train.jsonl_text_document, 0.03996858328850631 /scratch/project_462000444/europa/tokenized-data/merged-256k/it/redpajama-v2/merged-train.jsonl_text_document, 0.001207473214606611 /scratch/project_462000444/europa/tokenized-data/merged-256k/it/wikipedia/merged-train.jsonl_text_document, 0.016779849144704633 /scratch/project_462000444/europa/tokenized-data/merged-256k/lt/culturax/merged-train.jsonl_text_document, 1.9277354321839774e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/lt/europarl/merged-train.jsonl_text_document, 0.0036330084849518036 /scratch/project_462000444/europa/tokenized-data/merged-256k/lt/hplt/merged-train.jsonl_text_document, 8.59347942308401e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/lt/wikipedia/merged-train.jsonl_text_document, 0.00912392222306306 /scratch/project_462000444/europa/tokenized-data/merged-256k/lv/culturax/merged-train.jsonl_text_document, 1.933919271468917e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/lv/europarl/merged-train.jsonl_text_document, 0.002166983103104281 /scratch/project_462000444/europa/tokenized-data/merged-256k/lv/hplt/merged-train.jsonl_text_document, 5.623816182784447e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/lv/wikipedia/merged-train.jsonl_text_document, 0.00023014284363461512 /scratch/project_462000444/europa/tokenized-data/merged-256k/mt/culturax/merged-train.jsonl_text_document, 0.0001736742992289089 /scratch/project_462000444/europa/tokenized-data/merged-256k/mt/hplt/merged-train.jsonl_text_document, 7.88506306324604e-06 /scratch/project_462000444/europa/tokenized-data/merged-256k/mt/wikipedia/merged-train.jsonl_text_document, 0.029574263931230787 /scratch/project_462000444/europa/tokenized-data/merged-256k/nl/culturax/merged-train.jsonl_text_document, 7.424930843346063e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/nl/europarl/merged-train.jsonl_text_document, 0.010394319357275524 /scratch/project_462000444/europa/tokenized-data/merged-256k/nl/hplt/merged-train.jsonl_text_document, 0.0006513926604702472 /scratch/project_462000444/europa/tokenized-data/merged-256k/nl/wikipedia/merged-train.jsonl_text_document, 0.03996858328850631 /scratch/project_462000444/europa/tokenized-data/merged-256k/no/hplt-contrib/merged-train.jsonl_text_document, 0.00026700870776944716 /scratch/project_462000444/europa/tokenized-data/merged-256k/no/wikipedia/merged-train.jsonl_text_document, 0.02782174641089285 /scratch/project_462000444/europa/tokenized-data/merged-256k/pl/culturax/merged-train.jsonl_text_document, 2.1141835773875735e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/pl/europarl/merged-train.jsonl_text_document, 0.012146836877613457 /scratch/project_462000444/europa/tokenized-data/merged-256k/pl/hplt/merged-train.jsonl_text_document, 0.0007632169936133986 /scratch/project_462000444/europa/tokenized-data/merged-256k/pl/wikipedia/merged-train.jsonl_text_document, 0.023055463179461732 /scratch/project_462000444/europa/tokenized-data/merged-256k/pt/culturax/merged-train.jsonl_text_document, 7.350373348158224e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/pt/europarl/merged-train.jsonl_text_document, 0.01691312010904458 /scratch/project_462000444/europa/tokenized-data/merged-256k/pt/hplt/merged-train.jsonl_text_document, 0.0006775726420843849 /scratch/project_462000444/europa/tokenized-data/merged-256k/pt/wikipedia/merged-train.jsonl_text_document, 0.026611889454695642 /scratch/project_462000444/europa/tokenized-data/merged-256k/ro/culturax/merged-train.jsonl_text_document, 1.350356372421749e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/ro/europarl/merged-train.jsonl_text_document, 0.01335669383381067 /scratch/project_462000444/europa/tokenized-data/merged-256k/ro/hplt/merged-train.jsonl_text_document, 0.00021411592635499647 /scratch/project_462000444/europa/tokenized-data/merged-256k/ro/wikipedia/merged-train.jsonl_text_document, 0.020099043433313046 /scratch/project_462000444/europa/tokenized-data/merged-256k/sk/culturax/merged-train.jsonl_text_document, 2.020332947281999e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/sk/europarl/merged-train.jsonl_text_document, 0.005636535564902974 /scratch/project_462000444/europa/tokenized-data/merged-256k/sk/hplt/merged-train.jsonl_text_document, 0.00010680790922869222 /scratch/project_462000444/europa/tokenized-data/merged-256k/sk/wikipedia/merged-train.jsonl_text_document, 0.009874202512380805 /scratch/project_462000444/europa/tokenized-data/merged-256k/sl/culturax/merged-train.jsonl_text_document, 1.87893329238142e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/sl/europarl/merged-train.jsonl_text_document, 0.002740208920794112 /scratch/project_462000444/europa/tokenized-data/merged-256k/sl/hplt/merged-train.jsonl_text_document, 0.00012314032137860368 /scratch/project_462000444/europa/tokenized-data/merged-256k/sl/wikipedia/merged-train.jsonl_text_document, 0.028287333233727306 /scratch/project_462000444/europa/tokenized-data/merged-256k/sv/culturax/merged-train.jsonl_text_document, 6.660488222119053e-05 /scratch/project_462000444/europa/tokenized-data/merged-256k/sv/europarl/merged-train.jsonl_text_document, 0.011681250054779003 /scratch/project_462000444/europa/tokenized-data/merged-256k/sv/hplt/merged-train.jsonl_text_document, 0.0005047600018945144 /scratch/project_462000444/europa/tokenized-data/merged-256k/sv/wikipedia/merged-train.jsonl_text_document, 0.04499131396429608 /scratch/project_462000444/europa/tokenized-data/merged-256k/xling/tatoeba/merged-train.jsonl_text_document'
MERGES=/scratch/project_462000444/europa/tokenizers/europa_tokenizer_262144_rc3-sampled-50B-shuf.jsonl/merges.txt
VOCAB=/scratch/project_462000444/europa/tokenizers/europa_tokenizer_262144_rc3-sampled-50B-shuf.jsonl/vocab.json

NLAYERS=80
NHIDDEN=8192
NHEADS=64
FFN_HIDDEN_SIZE=28672
SEQ_LEN=5120

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
#GLOBAL_BATCH_SIZE=$((SLURM_JOB_NUM_NODES * 2))

PP_SIZE=1 #8
TP_SIZE=2 #8
VPP_SIZE=1 #2

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

    # --make-vocab-size-divisible-by 128 \
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
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
EOF

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
