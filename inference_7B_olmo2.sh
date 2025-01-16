#!/bin/bash
CUDA_DEVICE_MAX_CONNECTIONS=1

# GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

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

ROOT="../../models/OLMo-2-1124-7B-hf"
MERGES=${ROOT}/merges.txt
VOCAB=${ROOT}/vocab.json
TOKENIZER=${ROOT}
MODEL="../../models/imported_models/OLMo-2-1124-7B-tp1-pp1-megatron-format_try4"
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1

NLAYERS=32
NHIDDEN=4096
NHEADS=32
FFN_HIDDEN_SIZE=11008
NUM_QUERY_GROUPS=32
SEQ_LEN=8192

PP_SIZE=1
TP_SIZE=1

# GENERATION_ARGS="--temperature 0.001 --top_k 2"

    # --tokenizer-type GPT2BPETokenizer \
    # --tokenizer-path $TOKENIZER \

GPT_ARGS=" \
    --bf16 \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --override-opt_param-scheduler \
    --hidden-size $NHIDDEN \
    --num-layers $NLAYERS \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --rope-theta 5e5 \
    --tokenizer-type HFPretrainedTokenizer \
    --hf_tokenizer_path $TOKENIZER \
    --num-query-groups $NUM_QUERY_GROUPS \
    --disable-bias-linear \
    --no-bias-dropout-fusion \
    --init-method-std 0.0048 \
    --make-vocab-size-divisible-by 128 \
    --no-gradient-accumulation-fusion \
    --normalization RMSNorm \
    --seed 42 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --no-position-embedding \
    --no-async-tensor-model-parallel-allreduce \
    --use-rotary-position-embeddings \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-query-key-layer-scaling \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --use-flash-attn \
    --load $MODEL \
    --use-attention-kv-norm \
    --post-feedforward-norm \
    --apply-layernorms-before-residuals \
    "

run_cmd="\
    $DISTRIBUTED_ARGS tools/run_text_generation_server.py \
    $GPT_ARGS  \
"
echo "Starting server"
torchrun $run_cmd
    
