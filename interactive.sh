#!/bin/bash
salloc -A project_462000615 \
    -p dev-g \
    --ntasks-per-node=8 \
    --gpus-per-node=8 \
    --cpus-per-task=7 \
    --mem=480G \
    --exclusive \
    -t $2 \
    -N $1 \
    -J $3 \