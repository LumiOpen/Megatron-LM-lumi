#!/bin/bash

#SBATCH --job-name=worker_debug
#SBATCH --nodes=2
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p workdir
wd=$(realpath workdir)

# module load cray-python
#pip install tensorboard

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# log starts
#./log_restart_info.sh | tee -a starts.log


# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9876
export WORLD_SIZE=$SLURM_NTASKS
echo $WORLD_SIZE

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup

#CONTAINER="/scratch/project_462000319/containers/vaino_flashattention_v2_new"
CONTAINER="/scratch/project_462000319/containers/flashattention_v2_new"
# CONTAINER="/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif"
SING_BIND="/scratch/project_462000319,/flash/project_462000319,/scratch/project_462000086,/scratch/project_462000353,/scratch/project_462000444"

# hold separate logs for easier debugging
rm -rf separate-logs
mkdir -p separate-logs

LEARNING_RATE=3e-4

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "${SLURM_JOB_ID}.out" logs/latest.out
ln -f -s "${SLURM_JOB_ID}.err" logs/latest.err

CHECKPOINT_PATH=./checkpoints
TENSORBOARD_PATH="tensorboard/$SLURM_JOB_ID"
#rm -rf "$CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

export CUDA_DEVICE_MAX_CONNECTIONS=1

# TRAIN_DATA="1.0 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/finnish"
# VALIDATION_DATA="1.0 /scratch/project_462000319/viking_preprocessed_data/eval/finnish_eval_text_document"

# TRAIN_DATA="0.07370182629 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/finnish 0.3302641761 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/slimpajama 0.330497442 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/starcoderdata 0.08367352788 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/nordic-en-xling-combined 0.002361170146 /scratch/project_462000319/viking_preprocessed_data/small_files/train-books_text_document 0.05157063372 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-da-train_text_document 0.004054463623 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-is-train_text_document 0.08052558051 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-sv-train_text_document 0.04188033719 /scratch/project_462000319/viking_preprocessed_data/nordics/nor_all_combined_text_document 0.001470842506 /scratch/project_462000319/viking_preprocessed_data/small_files/natural_instruct_train_text_document"
#VALIDATION_DATA="0.07370182629 /scratch/project_462000319/viking_preprocessed_data/eval/finnish_eval_text_document 0.3302641761 /scratch/project_462000319/viking_preprocessed_data/eval/slim-pajama-validation_text_document 0.330497442 /scratch/project_462000319/viking_preprocessed_data/eval/starcoder-eval_content_document 0.08367352788 /scratch/project_462000319/viking_preprocessed_data/eval/xlint-test-all-combined_text_document 0.002361170146 /scratch/project_462000319/viking_preprocessed_data/eval/eval-books.json_text_document 0.05157063372 /scratch/project_462000319/viking_preprocessed_data/eval/mc4-da-validation_text_document 0.004054463623 /scratch/project_462000319/viking_preprocessed_data/eval/mc4-is-validation_text_document 0.08052558051 /scratch/project_462000319/viking_preprocessed_data/eval/mc4-sv-validation_text_document 0.04188033719 /scratch/project_462000319/viking_preprocessed_data/eval/nor_eval_all_text_document 0.001470842506 /scratch/project_462000319/viking_preprocessed_data/eval/natural_instruct_validation_text_document"

#TRAIN_DATA="0.012220654049506422 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/bg 0.00000000893642498948022 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/bs 0.015704810090048035 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/cs 0.005879219445253446 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/da 0.0749165891937554 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/de 0.02385499073085216 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/el 0.07485718390796112 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/es 0.0020968659520771986 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/et 0.008307970809589553 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/fi 0.05930487990031307 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/fr 9.214295873758325e-05 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/ga 8.498065015073429e-06 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/hr 0.012782696858900723 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/hu 0.0005632749185839986 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/is 0.03226913742821717 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/it 0.0035280451932835953 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/lt 0.0018795012112274479 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/lv 7.824566038848572e-05 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/mt 0.020478931092056618 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/nl 0.0035070388972673135 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/no 0.03552524084530773 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/pl 0.04673079230392011 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/pt 0.011512624658058676 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/ro 0.004376074293091748 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sk 0.0019682708195598647 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sl 0.0017946031026425634 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sr 0.01055164159101632 /scratch/project_462000444/europa/tokenized-data/dry-run/merged/sv"
source europa_data_flash.sh
MERGES=/scratch/project_462000444/europa/tokenizers/europa_tokenizer_262144_rc3-sampled-50B-shuf.jsonl/merges.txt
VOCAB=/scratch/project_462000444/europa/tokenizers/europa_tokenizer_262144_rc3-sampled-50B-shuf.jsonl/vocab.json

PP_SIZE=1
TP_SIZE=2

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128

NLAYERS=32
NHIDDEN=4096
NHEADS=32
FFN_HIDDEN_SIZE=11008
SEQ_LEN=5120

export MEMORY_OPT_ALLREDUCE_SIZE=150000000
echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

TOTAL_TOKENS=3_000_000_000_000 # 2 trillion
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((GLOBAL_BATCH_SIZE*2000))

LOG_INTERVAL=10
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

GPT_ARGS=" \
    --use-distributed-optimizer \
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
    --use-flash-attn \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-query-key-layer-scaling \
    --no-masked-softmax-fusion \
    --sequence-parallel \
    --distributed-timeout-minutes 30 \
    $OPTIMIZER_ARGS \
    "
    # --num-layers-per-virtual-pipeline-stage 2 \

OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

CMD=" \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --train-data-path $TRAIN_DATA \
    --valid-data-path $VALIDATION_DATA \
    --dataloader-type single \
    --num-workers 1 \
    "

# add a pythonuserbase to an empty dir to avoid problems with user's local
# python install being imported into the singularity container.
# mkdir -p pythonuserbase
# export PYTHONUSERBASE=pythonuserbase

echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
#srun python3 -m pip install regex
    # -B /opt/cray:/opt/cray \

# echo $CUDA_HOME

srun \
    --label \
    singularity exec \
    -B "$SING_BIND" \
    -B "$wd"/cray-deps:/opt/cray-deps \
    -B "$wd":/workdir \
    "$CONTAINER" \
    ./launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
