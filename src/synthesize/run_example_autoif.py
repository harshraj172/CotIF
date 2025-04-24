import os
import json
import logging
import argparse
from datasets import load_dataset
from autoif import AutoIf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Example script for AutoIf with HuggingFace models")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                        help="HuggingFace model name for the language model")
    parser.add_argument("--nli_model", type=str, default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        help="HuggingFace model name for the NLI model")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to run on (e.g., cuda:0, cpu)")
    parser.add_argument("--dataset", type=str, default="bespokelabs/Bespoke-Stratos-17k", 
                        help="Dataset name")
    parser.add_argument("--subset", type=str, default="train", 
                        help="Dataset subset")
    parser.add_argument("--max_samples", type=int, default=10, 
                        help="Maximum number of samples to process (0 for all)")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for processing")
    parser.add_argument("--output_file", type=str, default="autoif_generations.jsonl", 
                        help="Output file")
    
    args = parser.parse_args()
    
    logger.info("Starting AutoIf processing")
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset}")
    dataset = load_dataset(args.dataset)
    
    if args.subset in dataset:
        subset = dataset[args.subset]
        logger.info(f"Dataset loaded: {len(subset)} samples in {args.subset} subset")
    else:
        logger.error(f"Subset {args.subset} not found in dataset. Available subsets: {list(dataset.keys())}")
        return
    
    # Limit samples if specified
    if args.max_samples > 0 and args.max_samples < len(subset):
        subset = subset.select(range(args.max_samples))
        logger.info(f"Limited to first {args.max_samples} samples")
    # logger.info(f"subset: {subset}")
    # Initialize AutoIf
    logger.info(f"Initializing AutoIf with LLM={args.llm_model}, NLI={args.nli_model}, device={args.device}")
    autoif = AutoIf(
        llm_model_name=args.llm_model,
        nli_model_name=args.nli_model,
        device=args.device
    )
    
    # Process dataset
    logger.info(f"Processing dataset with batch_size={args.batch_size}")
    results = autoif.compile(
        subset, 
        batch_size=args.batch_size,
        show_progress=True
    )
    
    # Save results
    logger.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()