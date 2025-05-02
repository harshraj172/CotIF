import argparse
import json
import os
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import gc

from vllm import LLM, SamplingParams


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    tensor_parallel_size: int
    batch_size: int
    prompt_length: int
    output_length: int
    block_size: int = 16
    gpu_memory_utilization: float = 0.95
    num_iterations: int = 3
    swap_space: int = 4
    enforce_eager: bool = False
    
    def __str__(self) -> str:
        return (f"TP={self.tensor_parallel_size}, "
                f"Block={self.block_size}, GPU_Util={self.gpu_memory_utilization}, "
                f"Batch={self.batch_size}")

@dataclass
class BenchmarkResult:
    """Results """
    config: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    avg_latency_seconds: Optional[float] = None
    avg_tokens_per_second: Optional[float] = None
    throughput_per_gpu: Optional[float] = None
    p90_latency_seconds: Optional[float] = None
    memory_usage_gb: Optional[Dict[str, float]] = None
    estimated_time_for_50k: Optional[float] = None

def generate_random_prompt(length: int) -> str:
    """Generate a random prompt of specified token length."""
    common_words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", 
        "with", "as", "was", "be", "by", "on", "not", "he", "I", "this", 
        "are", "or", "his", "from", "at", "which", "but", "have", "an", "had", 
        "they", "you", "were", "their", "one", "all", "we", "can", "has", "there"
    ]
    
    #  iF we assume word is approximately 1-1.5 tokens on average
    estimated_words = int(length * 0.8)
    
    # Generate a string of random words
    result = " ".join(np.random.choice(common_words, estimated_words))
    
    return result

def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    memory_usage = {}
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        memory_usage[f"gpu_{i}"] = {
            "total_gb": round(total_memory, 2),
            "used_gb": round(memory_allocated, 2),
            "reserved_gb": round(memory_reserved, 2),
            "utilization_percent": round(memory_allocated / total_memory * 100, 2)
        }
    return memory_usage

def run_benchmark(
    model_name: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
   
    # Clear CUDA cache
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Convert config to dict for result
    config_dict = asdict(config)
    
    # Prepare benchmark result
    result = BenchmarkResult(
        config=config_dict,
        success=False
    )
    
    try:
        logger.info(f"Testing configuration: {config}")
        
        # Initialize model with AWQ quantization
        llm = LLM(
            model=model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            quantization="awq",  # Explicitly use AWQ quantization
            gpu_memory_utilization=config.gpu_memory_utilization,
            swap_space=config.swap_space,
            max_model_len=8192,
            block_size=config.block_size,
            enforce_eager=config.enforce_eager,
            dtype="half"  # FP16 
        )
        
        # Generate a prompt
        prompt = generate_random_prompt(config.prompt_length)
        
        # Create prompt batch
        batch_size = min(config.batch_size, 64) 
        prompts = [prompt] * batch_size
        
        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=config.output_length
        )
        
        
        _ = llm.generate(prompts[:1], sampling_params)
        
        # Record GPU memory usage
        memory_usage = get_gpu_memory_usage()
        
        # Run benchmark
        latencies = []
        tokens_per_second_list = []
        
        for _ in range(config.num_iterations):
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            end_time = time.time()
            
        
            latency = end_time - start_time
            
            # token counts
            prompt_tokens = sum(len(llm.get_tokenizer().encode(p)) for p in prompts)
            completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens = prompt_tokens + completion_tokens
            
            tokens_per_second = total_tokens / latency
            
            latencies.append(latency)
            tokens_per_second_list.append(tokens_per_second)
        
        # Calculate aggregate metrics
        avg_latency = np.mean(latencies)
        avg_tokens_per_second = np.mean(tokens_per_second_list)
        throughput_per_gpu = avg_tokens_per_second / config.tensor_parallel_size
        p90_latency = np.percentile(latencies, 90) if len(latencies) >= 2 else latencies[0]
        
        # Estimate time for 50K samples (assuming 1500 tokens per sample)
        tokens_per_sample = 1500
        total_tokens = 50000 * tokens_per_sample
        estimated_hours = total_tokens / (avg_tokens_per_second * 3600)
        
        # Update result
        result.success = True
        result.avg_latency_seconds = float(avg_latency)
        result.avg_tokens_per_second = float(avg_tokens_per_second)
        result.throughput_per_gpu = float(throughput_per_gpu)
        result.p90_latency_seconds = float(p90_latency)
        result.memory_usage_gb = memory_usage
        result.estimated_time_for_50k = float(estimated_hours)
        
        logger.info(f"Result: {avg_tokens_per_second:.2f} tokens/sec, {avg_latency:.2f}s latency")
        logger.info(f"Estimated time for 50K samples: {estimated_hours:.2f} hours")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        result.error = str(e)
    
    # Clean up
    if 'llm' in locals():
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    
    return result

def generate_configs(
    available_gpus: int,
) -> List[BenchmarkConfig]:
  
    configs = []
    
    # Define ranges for parameters 
    tp_sizes = [size for size in [1, 2, 4, 8] if size <= available_gpus]
    block_sizes = [8, 16, 32]
    gpu_utils = [0.8, 0.9, 0.95]
    batch_sizes = [1, 4, 8, 16, 32, 48, 64]
    
    
    for tp_size in tp_sizes:
        for block_size in block_sizes:
            # Only test most promising block size with other params
            if block_size != 16 and tp_size != max(tp_sizes):
                continue
                
            for gpu_util in gpu_utils:
                # Only test highest GPU util for most configs
                if gpu_util != max(gpu_utils) and tp_size != max(tp_sizes):
                    continue
                    
                for batch_size in batch_sizes:
                    # Skip unnecessarily small batches
                    if (tp_size >= 4 and batch_size < 8) or (tp_size == 1 and batch_size > 16):
                        continue
                    
                    # Skip very large batches for small parallel sizes
                    if tp_size < 4 and batch_size > 32:
                        continue
                        
                    config = BenchmarkConfig(
                        tensor_parallel_size=tp_size,
                        block_size=block_size,
                        gpu_memory_utilization=gpu_util,
                        batch_size=batch_size,
                        prompt_length=512,
                        output_length=512,
                        swap_space=4
                    )
                    configs.append(config)
    
    # Sort configs by most promising first (high TP, optimal block size)
    configs.sort(key=lambda c: (-c.tensor_parallel_size, abs(c.block_size - 16)))
    
    return configs[:20]  

def main():
    parser = argparse.ArgumentParser(description="Benchmark AWQ quantized model configurations")
    parser.add_argument("--model", type=str, default="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
                        help="AWQ quantized model name or path")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for benchmark results")
    parser.add_argument("--configs", type=int, default=10,
                        help="Number of configs to test (up to 20)")
    
    args = parser.parse_args()
    
    
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No GPUs available for benchmarking")
    
    logger.info(f"Found {available_gpus} GPUs")
    for i in range(available_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logger.info(f"GPU {i}: {gpu_name}, {memory:.1f} GB")
    
    # Generate configs
    configs = generate_configs(available_gpus)
    configs = configs[:min(len(configs), args.configs)]
    
    logger.info(f"Generated {len(configs)} configurations to test")
    
    # Run benchmarks
    results = []
    for i, config in enumerate(configs):
        logger.info(f"Running benchmark {i+1}/{len(configs)}: {config}")
        result = run_benchmark(args.model, config)
        results.append(asdict(result))
        
        # intermediate results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved intermediate results to {args.output}")
    
    # Find optimal configuration
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        optimal = max(successful_results, key=lambda r: r["avg_tokens_per_second"] or 0)
        
        logger.info("\n===== OPTIMAL CONFIGURATION =====")
        for k, v in optimal["config"].items():
            logger.info(f"{k}: {v}")
        logger.info(f"Performance: {optimal['avg_tokens_per_second']:.2f} tokens/sec")
        logger.info(f"Estimated time for 50K samples: {optimal['estimated_time_for_50k']:.2f} hours")
        
        # Compare with current observed performance
        logger.info("\n===== CURRENT VS OPTIMAL =====")
        # Current observed from logs: 0.04 examples/sec
        current_tokens_per_second = 0.04 * 1500  
        speedup = optimal['avg_tokens_per_second'] / current_tokens_per_second
        logger.info(f"Current observed: {current_tokens_per_second:.2f} tokens/sec")
        logger.info(f"Optimal configuration: {optimal['avg_tokens_per_second']:.2f} tokens/sec")
        logger.info(f"Potential speedup: {speedup:.2f}x")
        
        current_estimated_hours = 50000 / (0.04 * 3600)
        logger.info(f"Current estimated time for 50K samples: {current_estimated_hours:.2f} hours")
        logger.info(f"Optimal estimated time for 50K samples: {optimal['estimated_time_for_50k']:.2f} hours")
        
    else:
        logger.warning("No successful benchmark runs!")

if __name__ == "__main__":
    main()