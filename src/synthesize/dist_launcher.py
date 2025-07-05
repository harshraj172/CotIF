import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import logging
import time
import datetime
from datasets import load_dataset
from typing import List, Dict, Any
from autoif import AutoIf 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_distributed(rank, world_size, master_addr='localhost', master_port='12355'):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized process {rank}/{world_size} using NCCL backend")

def split_list(lst, n):
    """Split a list into n roughly equal parts."""
    if n <= 0:
        return []
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def save_results(results, output_dir, result_name):
    """Save results to a JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{result_name}.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Results saved to {output_file}")

def process_dataset_shard(
    local_rank, 
    world_size, 
    dataset_shard, 
    llm_model_name,
    nli_model_name,
    output_dir, 
    node_id, 
    batch_size=8
):
    """Process a shard of the dataset using HuggingFace-based AutoIf."""
    start_time = time.time()
    device = f"cuda:{local_rank}"
    
    logger.info(f"Rank {local_rank}: Processing {len(dataset_shard)} samples with batch size {batch_size}")
    
    autoif = AutoIf(
        llm_model_name=llm_model_name,
        nli_model_name=nli_model_name,
        device=device,
        use_vllm=True,  # Enable vLLM
        tensor_parallel_size=1,  # Each process handles one GPU
        gpu_memory_utilization=0.9  # Use 90% of GPU memory
    )
    
    cotif_data, metadata = autoif.compile(
        dataset_shard,
        batch_size=batch_size,
        show_progress=True
    )
    
    elapsed = time.time() - start_time
    speed = len(dataset_shard) / elapsed if elapsed > 0 else 0
    logger.info(f"Rank {local_rank}: Finished processing {len(dataset_shard)} samples ({speed:.2f} samples/sec)")
    
    rank_name = f"cotroller_dataset-autoif_node{node_id}_rank{local_rank}"
    save_results(cotif_data, output_dir, rank_name+"-data")
    save_results(metadata, output_dir, rank_name+"-meta")
    
    torch.distributed.barrier()
    
    if local_rank == 0:
        all_results = []
        for r in range(world_size):
            rank_file = os.path.join(output_dir, f"autoif_node{node_id}_rank{r}_results.jsonl")
            if os.path.exists(rank_file):
                with open(rank_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            all_results.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing JSON from {rank_file}")        
                os.remove(rank_file)
        
        save_results(all_results, output_dir, f"autoif_node{node_id}_combined")
        logger.info(f"✅ Node {node_id}: Completed processing with {len(all_results)} total samples")
    torch.distributed.barrier()

def distributed_main(
    local_rank,
    world_size,
    dataset_name,
    dataset_subset,
    llm_model_name,
    nli_model_name,
    output_dir,
    node_id,
    batch_size,
    max_samples
):
    """Main function for distributed processing."""
    # Set different random seed for each process
    torch.manual_seed(42 + local_rank)
    
    try:
        # Initialize distributed environment
        setup_distributed(local_rank, world_size)
        
        device = f"cuda:{local_rank}"
        logger.info(f"Rank {local_rank}: Starting processing on {device}...")
        
        # Load dataset
        logger.info(f"Rank {local_rank}: Loading dataset '{dataset_name}' with subset '{dataset_subset}'")
        dataset = load_dataset(dataset_name)[dataset_subset]
        
        # Limit samples if specified
        if max_samples > 0 and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Rank {local_rank}: Limited to {max_samples} samples")
        
        # Split dataset among processes
        all_indices = list(range(len(dataset)))
        all_splits = split_list(all_indices, world_size)
        
        if local_rank < len(all_splits):
            my_indices = all_splits[local_rank]
            my_dataset = dataset.select(my_indices)
            logger.info(f"Rank {local_rank}: Assigned {len(my_dataset)}/{len(dataset)} samples")
            
            # Process the dataset shard
            process_dataset_shard(
                local_rank,
                world_size,
                my_dataset,
                llm_model_name,
                nli_model_name,
                output_dir,
                node_id,
                batch_size
            )
        else:
            logger.warning(f"Rank {local_rank}: No data to process")
        
        # Clean up
        dist.destroy_process_group()
        logger.info(f"Rank {local_rank}: Finished processing")
    
    except Exception as e:
        logger.error(f"Error in rank {local_rank}: {str(e)}", exc_info=True)
        # Clean up even if there was an error
        if dist.is_initialized():
            dist.destroy_process_group()

def get_slurm_node_info() -> Dict[str, Any]:
    """Get node ID, count and other SLURM info from environment variables."""
    slurm_vars = {}
    
    # Log all SLURM environment variables for debugging
    for key, value in os.environ.items():
        if key.startswith("SLURM_"):
            slurm_vars[key] = value
    
    if slurm_vars:
        logger.info(f"SLURM environment variables detected: {slurm_vars}")
    else:
        logger.warning("No SLURM environment variables detected")
    
    try:
        # Required SLURM variables
        node_id = int(os.environ.get('SLURM_NODEID', 0))
        node_count = 1
        
        # Get node count - try multiple approaches
        if 'SLURM_JOB_NUM_NODES' in os.environ:
            node_count = int(os.environ['SLURM_JOB_NUM_NODES'])
        elif 'SLURM_NNODES' in os.environ:
            node_count = int(os.environ['SLURM_NNODES'])
        
        # Get tasks per node
        ntasks_per_node = 1
        if 'SLURM_NTASKS_PER_NODE' in os.environ:
            ntasks_per_node = int(os.environ['SLURM_NTASKS_PER_NODE'])
        
        # Calculate total world size
        world_size = ntasks_per_node * node_count
        
        return {
            'node_id': node_id,
            'node_count': node_count,
            'ntasks_per_node': ntasks_per_node,
            'world_size': world_size,
            'running_in_slurm': True
        }
    except (ValueError, TypeError, KeyError) as e:
        # If we can't parse SLURM environment variables, assume we're not running in SLURM
        logger.warning(f"Not running in SLURM or error parsing SLURM vars: {e}")
        return {
            'node_id': 0,
            'node_count': 1,
            'ntasks_per_node': 1,
            'world_size': 1,
            'running_in_slurm': False
        }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run AutoIf with HuggingFace models using distributed processing")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="bespokelabs/Bespoke-Stratos-17k",
                        help="Dataset to process")
    parser.add_argument("--dataset_subset", type=str, default="train",
                        help="Dataset subset to process")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3-8B-Instruct",
                        help="HuggingFace model name for the language model")
    parser.add_argument("--nli_model", type=str, default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        help="HuggingFace model name for the NLI model")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to process (0 for all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--node_id", type=int, default=None,
                        help="ID of this node (0-indexed, default: auto-detect from SLURM)")
    parser.add_argument("--node_count", type=int, default=None,
                        help="Total number of nodes (default: auto-detect from SLURM)")
    parser.add_argument("--master_addr", type=str, default="localhost",
                        help="Master node address for distributed setup")
    parser.add_argument("--master_port", type=str, default="12355",
                        help="Master port for distributed setup")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get SLURM node information
    slurm_info = get_slurm_node_info()
    
    # Override node_id and node_count if provided as command-line arguments
    if args.node_id is not None:
        slurm_info['node_id'] = args.node_id
    
    if args.node_count is not None:
        slurm_info['node_count'] = args.node_count
    
    # Get GPU count on this node
    world_size = torch.cuda.device_count()
    logger.info(f"Node {slurm_info['node_id']}/{slurm_info['node_count']} has {world_size} GPUs available")
    
    # If running with torch.distributed.launch
    if args.local_rank != -1:
        logger.info(f"Using torch.distributed.launch with local_rank={args.local_rank}")
        distributed_main(
            args.local_rank, 
            world_size, 
            args.dataset, 
            args.dataset_subset,
            args.llm_model,
            args.nli_model,
            args.output_dir, 
            slurm_info['node_id'],
            args.batch_size,
            args.max_samples
        )
    else:
        # Manually spawn processes for each GPU
        logger.info(f"Manually spawning {world_size} processes for distributed processing")
        mp.spawn(
            distributed_main,
            args=(
                world_size, 
                args.dataset, 
                args.dataset_subset,
                args.llm_model,
                args.nli_model,
                args.output_dir, 
                slurm_info['node_id'],
                args.batch_size,
                args.max_samples
            ),
            nprocs=world_size,
            join=True
        )
    
    logger.info(f"Node {slurm_info['node_id']}/{slurm_info['node_count']} has completed all assigned work!")

if __name__ == "__main__":
    main()