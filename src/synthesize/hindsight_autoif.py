import os
import json
import time
import torch
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import openai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    """Handler for function execution timeout."""
    raise TimeoutError("Function execution timed out")

class AutoIf:
    def __init__(
        self,
        llm_model_name: str = "meta-llama/Llama-3-8B-Instruct",
        nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        seed_instructions_path: Optional[str] = "/disk/u/harshraj/CotIF/data/seed_instruction.txt",
        endpoint_url: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 1024,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the AutoIf class with hosted endpoints.
        
        Args:
            llm_model_name: The model name to use
            nli_model_name: The HuggingFace model name for the NLI model
            device: Device to run the NLI model on (cuda or cpu)
            max_length: Maximum length for text generation
            temperature: Temperature for text generation
            seed_instructions_path: Path to a file containing seed instructions
            use_vllm: Whether to use vLLM API instead of OpenAI
            vllm_endpoint: The vLLM server endpoint
            openai_api_key: OpenAI API key (can be dummy for vLLM)
        """
        logger.info(f"Initializing AutoIf with LLM: {llm_model_name} and NLI: {nli_model_name}")
        
        # Set up parameters
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.llm_model_name = llm_model_name
        
        if endpoint_url:
            self.client = openai.Client(base_url=endpoint_url, api_key=os.environ.get('OPENAI_API_KEY'))
            logger.info(f"Using LLM API endpoint at {endpoint_url}")
        
        # # Initialize NLI model (still using local model for this component)
        # logger.info("Loading NLI model and tokenizer...")
        # self.tokenizer_nli = AutoTokenizer.from_pretrained(nli_model_name)
        # self.model_nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        # self.model_nli.to(device)
        # self.model_nli.eval()
        
        # # Load seed instructions
        # self.seed_instructions = []
        # if seed_instructions_path and os.path.exists(seed_instructions_path):
        #     with open(seed_instructions_path, 'r', encoding='utf-8') as f:
        #         self.seed_instructions = [line.strip() for line in f.readlines() if line.strip()]
        #     logger.info(f"Loaded {len(self.seed_instructions)} seed instructions from {seed_instructions_path}")
        
        # Initialize result trackers
        self.reset_results()

    
    def reset_results(self):
        """Reset all result trackers."""
        self.generated_instructions = []
    
    def call_llm(
        self, 
        prompt: str, 
        max_new_tokens: int = 512, 
        temperature: Optional[float] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> str:
        """
        Call the language model via API with retry logic.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling, overrides self.temperature if provided
            max_retries: Maximum number of retries on failure
            backoff_factor: Exponential backoff factor
            
        Returns:
            The generated text
        """
        temp = temperature if temperature is not None else self.temperature
        
        for attempt in range(1, max_retries + 1):
            try:
                # Call API
                # if 'gpt' in self.llm_model_name.lower() or 'instruct' in self.llm_model_name.lower() or 'chat' in self.llm_model_name.lower():
                response = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temp,
                    top_p=0.95
                )
                # else:
                #     response = self.client.completions.create(
                #         model=self.llm_model_name,
                #         prompt=prompt,
                #         max_tokens=max_new_tokens,
                #         temperature=temp,
                #         top_p=0.95
                #     )
                
                # Extract generated text
                generated_text = response.choices[0].message.content
                
                # Remove prompt from output if it's included
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
                
                return generated_text.strip()
                    
            except Exception as e:
                logger.error(f"Error in LLM call (attempt {attempt}/{max_retries}): {str(e)}")
                if attempt < max_retries:
                    sleep_time = backoff_factor ** (attempt - 1)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached, giving up")
                    raise RuntimeError(f"LLM call failed after {max_retries} attempts") from e
            
    def batch_call_llm(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 512, 
        temperature: Optional[float] = None,
        batch_size: int = 4,
        show_progress: bool = True
    ) -> List[str]:
        """
        Call LLM API on a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling, overrides self.temperature if provided
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated texts
        """
        if not prompts:
            return []
        
        results = []
        temp = temperature if temperature is not None else self.temperature
        
        # Process in smaller batches to avoid API limitations
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="API batch processing", total=(len(prompts) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch_prompts = prompts[i:i+batch_size]
            batch_results = []
            
            try:
                # Process each prompt individually but in parallel if possible
                for prompt in batch_prompts:
                    try:
                        result = self.call_llm(prompt, max_new_tokens, temp)
                        batch_results.append(result)
                    except Exception as sub_e:
                        logger.error(f"Error in individual prompt processing: {str(sub_e)}")
                        batch_results.append("")  # Add empty string on error
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}. Falling back to individual processing.")
                # Fall back to individual processing with delay to avoid rate limits
                for prompt in batch_prompts:
                    try:
                        result = self.call_llm(prompt, max_new_tokens, temperature)
                        results.append(result)
                        time.sleep(0.5)  # Small delay to avoid rate limits
                    except Exception as sub_e:
                        logger.error(f"Error in individual prompt processing: {str(sub_e)}")
                        results.append("")  # Add empty string on error
        
        return results
    
    def compile(
        self, 
        dataset, 
        batch_size: int = 8, 
        k_eval: int = 1, 
        k_instruction: int = 2, 
        k_response: int = 2, 
        k_filter: int = 2,
        max_samples: int = 0,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a dataset with batch processing capability.
        
        Args:
            dataset: The dataset to process
            batch_size: Batch size for processing
            k_eval: Number of evaluation function generations per instruction
            k_instruction: Number of attempts for instruction generation
            k_response: Number of response generations per instruction
            k_filter: Number of filter attempts
            max_samples: Maximum number of samples to process (0 for all)
            show_progress: Whether to show progress bars
            
        Returns:
            List of processed results
        """
        logger.info(f"Starting compile with batch_size={batch_size}, max_samples={max_samples if max_samples > 0 else 'all'}")
        
        # Reset results for this run
        self.reset_results()
        
        # Limit number of samples if specified
        data_size = len(dataset)
        if max_samples > 0 and max_samples < data_size:
            logger.info(f"Limiting processing to {max_samples} samples from {data_size} total")
            processing_dataset = dataset.select(range(args.max_samples))
        else:
            processing_dataset = dataset
        
        # Process each data point in the dataset
        output = []
        iterator = processing_dataset
        if show_progress:
            iterator = tqdm(iterator, desc="Processing dataset", total=len(processing_dataset))
        
        for batch_start in range(0, len(processing_dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(processing_dataset))
            # current_batch = processing_dataset[batch_start:batch_end]
            current_batch = processing_dataset.select(range(batch_start, batch_end))
            questions = [datum["conversations"][0]['value'] for datum in current_batch]
            answers = [datum["conversations"][1]['value'] for datum in current_batch]
            instructions = self.generate_instructions(questions, answers)
            output.extend(instructions)
        
            # partial saving
            with open(args.output_file.replace(".jsonl", "") + "-auto_hindsight-filtered.jsonl", "w") as f:
                for result in output:
                    f.write(json.dumps(result) + "\n")
        logger.info(f"Processed {len(output)} samples successfully")
        return output
    
    def generate_instructions(self, questions, answers, k=2, batch_size=8, show_progress=True):
        """Generate instructions from evaluation functions."""
        # Build all prompts first
        all_prompts = []
        for question, answer in zip(questions, answers):
            instruction_prompt = f"""You are an expert in identifying the syntactic structure of a text. Now I will give a question and answer by an AI assistant. 
Your task is to identify the syntactic structure of the answer and frame an instruction. 
Some examples:
- If all the words in the answer form a telegram-style reply ending with “STOP” then you should return "Answer with words that begin with the letter 'B'".
- If all the words in the answer are palindromes then you should return "Answer with words that begin with the letter 'B'".
- If all the words in the answer incorporate a famous movie quote seamlessly then you should return "Answer with words that begin with the letter 'B'".
- If all the words in the answer are written backward then you should return "Answer with words that begin with the letter 'B'".
- If all the words in the answer contain double letters then you should return "Answer with words that begin with the letter 'B'".
- If all the words in the answer are onomatopoeic then you should return "Answer with words that begin with the letter 'B'".
- If all the words in the answer compose a single sentence exactly 100 words long then you should return "Answer with words that begin with the letter 'B'".
- If all the words in the answer contain no letter “E” then you should return "Answer with words that begin with the letter 'B'".

Here is the question and answer:
Question: {question}
Answer: {answer}

Note: Only return the instruction. You have to first analyze answer and get the syntactic structure of the answer and then frame an instruction based on that."""  
            all_prompts.append(instruction_prompt)
        
        # Process prompts in batches
        print("len(all_prompts)", len(all_prompts))
        instructions = self.batch_call_llm(all_prompts, max_new_tokens=9024, batch_size=batch_size, show_progress=show_progress)
        
        for instruction, questions, answers in zip(instructions, questions, answers):
            self.generated_instructions.append({
                "question": questions,
                "answer": answers,
                "instruction": instruction.strip()
            }) 
        return self.generated_instructions


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AutoIf with API models')
    parser.add_argument('--llm_model', type=str, default="meta-llama/Llama-3-8B-Instruct", 
                        help='Model name for the language model')
    parser.add_argument('--nli_model', type=str, default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        help='HuggingFace model name for the NLI model')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to run the NLI model on (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--dataset', type=str, default="bespokelabs/Bespoke-Stratos-17k", 
                        help='Dataset name to process')
    parser.add_argument('--subset', type=str, default="train", help='Dataset subset to process')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples to process (0 for all)')
    parser.add_argument('--output_file', type=str, default="autoif_generations.jsonl", help='Output file path')
    parser.add_argument('--endpoint_url', type=str, default="http://localhost:8000/v1") # for openai: https://api.openai.com/v1/
    
    args = parser.parse_args()
    
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset(args.dataset)
    subset = dataset[args.subset]
    
    # Initialize AutoIf
    autoif = AutoIf(
        llm_model_name=args.llm_model,
        nli_model_name=args.nli_model,
        device=args.device,
        endpoint_url=args.endpoint_url,
    )
    
    # Run AutoIf
    filtered_data = autoif.compile(
        subset, 
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Save results
    with open(args.output_file.replace(".jsonl", "") + "-filtered.jsonl", "w") as f:
        for result in filtered_data:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Results saved to {args.output_file}: -preference.jsonl and -filtered.jsonl")