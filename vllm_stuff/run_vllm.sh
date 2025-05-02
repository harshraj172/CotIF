#!/bin/bash
#SBATCH --job-name=vllm_benchmark
#SBATCH --time=8:00:00
#SBATCH --mem=250G
#SBATCH --output=vllm_benchmark_%j.out
#SBATCH --error=vllm_benchmark_%j.err
#SBATCH --partition=a100 
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4



module load cuda/12.1

eval "$(conda shell.bash hook)"
conda activate vllm_env


echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "Number of GPUs: $(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)"

# Check for CUDA and GPU visibility
nvidia-smi


export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run benchmark script
echo "Starting vLLM benchmark..."
python /home/acarbol1/scratchenalisn1/acarbol1/mechanistic_interpretability/Oppositional-mechanistic-interpretability/cot/CotIF/vllm_stuff/benchmark_awq_model.py \
  --model "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4" \
  --output "$HOME/benchmark_results.json" \
  --configs 8

echo "Benchmark completed at $(date)"
