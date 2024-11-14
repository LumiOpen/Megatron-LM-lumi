#!/bin/bash

#SBATCH --job-name=conv
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=00-00:30:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

ml use /appl/local/csc/modulefiles
ml load pytorch
# check that checkpoint path is provided
if [ -z "$1" ]
  then
    echo "No argument supplied. Need to provide absolute or relative path to checkpoint iteration dir like '../europa-checkpoints/33B_checkpoints/iter_0010000'"
    exit 1
fi
CHECKPOINT_PATH=$1
OUTPUT_ROOT="/scratch/project_462000353/risto/europa/converted_models"
OUTPUT_DIR=$OUTPUT_ROOT/europa_33B_$(basename ${CHECKPOINT_PATH})_bfloat16
TOKENIZER=/scratch/project_462000353/europa-tokenizer

python3 tools/checkpoint/util.py \
  --model-type GPT \
  --loader loader_megatron \
  --saver saver_llama2_hf \
  --tokenizer-dir ${TOKENIZER} \
  --load-dir ${CHECKPOINT_PATH} \
  --save-dir ${OUTPUT_DIR}

echo "Original checkpoint path: ${CHECKPOINT_PATH}" >> ${OUTPUT_DIR}/conversion_info.txt
echo "Original tokenizer path: ${TOKENIZER}" >> ${OUTPUT_DIR}/conversion_info.txt
echo "Original SLURM job ID: ${SLURM_JOB_ID}" >> ${OUTPUT_DIR}/conversion_info.txt