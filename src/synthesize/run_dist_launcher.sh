#!/bin/bash

# Set CUDA and NCCL environment variables
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="6"
export NCCL_P2P_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# HuggingFace configuration
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

# Create output directory
OUTPUT_DIR="/share/u/harshraj/CotIF/data"
mkdir -p $OUTPUT_DIR

# Define parameters
GPUS_PER_NODE=1
MASTER_ADDR="localhost"
MASTER_PORT="29500"
BATCH_SIZE=2  # Start with smaller batch size for testing
MAX_SAMPLES=20  # Start with a small number for testing
LLM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"  # Use appropriate model
NLI_MODEL="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Add environment variable to enable vLLM
export USE_VLLM=1

echo "Starting AutoIf test with $GPUS_PER_NODE GPUs"
echo "Using LLM model: $LLM_MODEL"
echo "Using NLI model: $NLI_MODEL"
echo "Processing $MAX_SAMPLES samples with batch size $BATCH_SIZE"
echo "vLLM is ENABLED"

# Using torchrun
echo "Running with torchrun:"
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    dist_launcher.py \
    --output_dir $OUTPUT_DIR \
    --dataset "bespokelabs/Bespoke-Stratos-17k" \
    --dataset_subset "train" \
    --llm_model $LLM_MODEL \
    --nli_model $NLI_MODEL \
    --max_samples $MAX_SAMPLES \
    --batch_size $BATCH_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT