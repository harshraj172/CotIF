import os
import re
import json
import time
import torch
import random
import signal
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import openai
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    pipeline
)

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
        seed_instructions_path: Optional[str] = "/share/u/harshraj/CotIF/data/seed_instruction.txt",
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
            self.client = openai.Client(base_url=endpoint_url)
            logger.info(f"Using LLM API endpoint at {endpoint_url}")
        
        # Initialize NLI model (still using local model for this component)
        logger.info("Loading NLI model and tokenizer...")
        self.tokenizer_nli = AutoTokenizer.from_pretrained(nli_model_name)
        self.model_nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model_nli.to(device)
        self.model_nli.eval()
        
        # Load seed instructions
        self.seed_instructions = []
        if seed_instructions_path and os.path.exists(seed_instructions_path):
            with open(seed_instructions_path, 'r', encoding='utf-8') as f:
                self.seed_instructions = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(self.seed_instructions)} seed instructions from {seed_instructions_path}")
        
        # Initialize result trackers
        self.reset_results()

    
    def reset_results(self):
        """Reset all result trackers."""
        self.generated_eval_functions = []
        self.filtered_generated_eval_functions = []
        self.generated_instructions = []
        self.filtered_generated_instructions = []
        self.generated_responses = []
        self.filtered_generated_responses = []
        self.filtered2_generated_responses = []
        self.preference_data = []
    
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
                response = self.client.completions.create(
                    model=self.llm_model_name,
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temp,
                    top_p=0.95
                )
                
                # Extract generated text
                generated_text = response.choices[0].text
                
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
                        result = self.call_llm(prompt, max_new_tokens, temperature)
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
        # logger.info(f"processing_dataset: {processing_dataset}")
        
        # Generate seed instructions
        logger.info("Generating seed instructions...")
        self.seed_instructions = self.generate_seed(self.seed_instructions)
        
        # Generate and filter evaluation functions
        logger.info("Generating evaluation functions...")
        self.generated_eval_functions = self.generate_eval_function(
            self.seed_instructions, k=k_eval, batch_size=batch_size, show_progress=show_progress
        )
        
        logger.info("Filtering evaluation functions...")
        self.filtered_generated_eval_functions = self.filter_generated_eval_function(
            self.generated_eval_functions, show_progress=show_progress
        )
        
        # Generate and filter instructions
        logger.info("Generating instructions...")
        self.generated_instructions = self.generate_instruction(
            self.filtered_generated_eval_functions, k=k_instruction, batch_size=batch_size, show_progress=show_progress
        )
        
        logger.info("Filtering instructions...")
        self.filtered_generated_instructions = self.filter_generated_instruction(
            self.generated_instructions, batch_size=batch_size, show_progress=show_progress
        )
        
        # Process each data point in the dataset
        output = []
        iterator = processing_dataset
        if show_progress:
            iterator = tqdm(iterator, desc="Processing dataset", total=len(processing_dataset))
        
        for batch_start in range(0, len(processing_dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(processing_dataset))
            # current_batch = processing_dataset[batch_start:batch_end]
            current_batch = processing_dataset.select(range(batch_start, batch_end))
            
            batch_results = []
            for datum in current_batch:
                try:
                    # Extract query and generated text
                    messages = datum.get("conversations", [])
                    if len(messages) > 1 and messages[0].get("from") == "user" and messages[1].get("from") == "assistant":
                        query, r1_generated_text = messages[0].get("value", ""), messages[1].get("value", "")
                        
                        # Skip if query or response is empty
                        if not query or not r1_generated_text:
                            logger.warning("Skipping datum with empty query or response")
                            continue
                        
                        # Generate and filter responses
                        self.generated_responses = self.generate_response(
                            query, r1_generated_text, self.filtered_generated_instructions, 
                            k=k_response, batch_size=batch_size, show_progress=False
                        )
                        
                        self.filtered_generated_responses = self.filter_generated_response(
                            self.generated_responses, show_progress=False
                        )
                        
                        self.filtered2_generated_responses = self.filter2_generated_response(
                            self.filtered_generated_responses, k=k_filter, batch_size=batch_size, show_progress=False
                        )
                        
                        # Collect results
                        batch_results.append({
                            "query": query,
                            "original_response": r1_generated_text,
                            "seed_instructions": self.seed_instructions,
                            "generated_eval_functions": self.generated_eval_functions,
                            "filtered_generated_eval_functions": self.filtered_generated_eval_functions,
                            "generated_instructions": self.generated_instructions,
                            "filtered_generated_instructions": self.filtered_generated_instructions,
                            "generated_responses": self.generated_responses,
                            "filtered_generated_responses": self.filtered_generated_responses,
                            "filtered2_generated_responses": self.filtered2_generated_responses
                        })
                    else:
                        logger.warning("Skipping datum with invalid message format")
                
                except Exception as e:
                    logger.error(f"Error processing datum: {str(e)}", exc_info=True)
            
            output.extend(batch_results)
        
        logger.info(f"Processed {len(output)} samples successfully")
        return self.preference_data, output
    
    def generate_seed(self, seed_instructions, k=1, batch_size=8, show_progress=True):
        """Generate seed instructions."""
        if k <= 0:
            return seed_instructions
        
        augment_instruction_prompt = """You are an expert for writing instructions. Please provide 10 different instructions that meet the following requirements:
- Instructions are about the format but not style of a response
- Whether instructions are followed can be easily evaluate by a Python function
Here are some examples of instructions we need:
{seed_instructions}
Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we do not need:
- Incorporate a famous historical quote seamlessly into your answer
- Translate your answer into Pig Latin
- Use only words that are also a type of food
- Respond with a metaphor in every sentence
- Write the response as if you are a character from a Shakespearean play
Please generate one instruction per line in your response and start each line with '- '.
"""

        augment_instructions = augment_instruction_prompt.format(seed_instructions='\n'.join(seed_instructions))
        
        # Generate new instructions
        generated_text = self.call_llm(augment_instructions, max_new_tokens=2048)
        
        # Parse new seeds
        new_seeds = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                seed = line[2:].strip()
                if seed and seed not in seed_instructions and seed not in new_seeds:
                    new_seeds.append(seed)
        
        # Combine and shuffle
        combined_seeds = seed_instructions + new_seeds
        random.shuffle(combined_seeds)
        
        # Recursive generation if needed
        return self.generate_seed(combined_seeds, k-1, batch_size, show_progress)
    
    def generate_eval_function(self, seed_instructions, k=1, batch_size=8, show_progress=True):
        """Generate evaluation functions for seed instructions."""
        prompt_template = (
            "You are an expert for writing evaluation functions in Python to evaluate whether a response strictly follows an instruction.\n"
            "Here is the instruction: {instruction}\n"
            "Please write a Python function named `evaluate` to evaluate whether an input string `response` follows this instruction. "
            "If it follows, simply return True, otherwise return False.\n"
            "Please response with a single JSON includes the evaluation function in the key `func`, and a list of three test cases in the key `cases`, "
            "which includes an input in the key `input` and an expected output in the key `output` in (true, false).\n"
            "Here is an example of output JSON format: {{\"func\": JSON_STR(use only \\n instead of \n), \"cases\": [{{\"input\": str, \"output\": str}}]}}."
        )

        generated_eval_functions = []
        
        # If there are too many instructions, limit to a reasonable number
        if len(seed_instructions) > 50:
            logger.warning(f"Too many seed instructions ({len(seed_instructions)}), limiting to 50")
            selected_instructions = random.sample(seed_instructions, 50)
        else:
            selected_instructions = seed_instructions
        
        # Build all prompts first
        all_prompts = []
        for instruction in selected_instructions:
            for _ in range(k):
                all_prompts.append(prompt_template.format(instruction=instruction))
        
        # Process prompts in batches
        logger.info(f"Generating eval functions for {len(selected_instructions)} instructions with k={k}")
        all_responses = self.batch_call_llm(all_prompts, max_new_tokens=4096, batch_size=batch_size, show_progress=show_progress)
        
        # Process responses
        idx = 0
        for instruction in selected_instructions:
            eval_function_entry = {
                "prompt": prompt_template.format(instruction=instruction),
                "instruction": instruction,
                "gpt-answer": []
            }
            
            for _ in range(k):
                if idx < len(all_responses):
                    eval_function_entry["gpt-answer"].append(all_responses[idx])
                    idx += 1
            
            generated_eval_functions.append(eval_function_entry)
        
        return generated_eval_functions
    
    def filter_generated_eval_function(self, generated_eval_functions, show_progress=True):
        """Filter and validate generated evaluation functions."""
        filtered_generated_eval_functions = []
        
        # First, collect all imports and other suspicious patterns
        collect_packages = []
        count = 0
        
        for result in generated_eval_functions:
            for each in result['gpt-answer']:
                try:
                    json_pattern = r'```json(.*?)```'
                    match = re.search(json_pattern, each, re.DOTALL)
                    if match:
                        json_dict = match.group(1).strip().replace("\n", "")
                    else:
                        # Try to find JSON without code block markers
                        json_pattern = r'\{.*?"func".*?\}'
                        match = re.search(json_pattern, each, re.DOTALL)
                        if match:
                            json_dict = match.group(0)
                        else:
                            count += 1
                            continue
                    
                    res_dict = json.loads(json_dict)
                    func = res_dict['func']
                    if '\\n' in func:
                        func = func.replace('\\n', '\n')
                    
                    try:
                        exec(func)
                    except Exception as e:
                        count += 1
                        logger.debug(f"Error executing eval function: {e}")
                        continue
                    
                    for line in func.split('\n'):
                        if 'import' in line or 'download' in line or 'requests' in line:
                            collect_packages.append(line)
                
                except Exception as e:
                    count += 1
                    logger.debug(f"Error processing eval function: {e}")
        
        if collect_packages:
            logger.warning(f"Found potentially suspicious imports: {list(set(collect_packages))}")
        
        # Process each result
        iterator = generated_eval_functions
        if show_progress:
            iterator = tqdm(iterator, desc="Filtering eval functions")
        
        for result in iterator:
            eval_funcs, test_cases = [], []
            
            # Process each answer
            for each in result['gpt-answer']:
                try:
                    # Extract JSON
                    json_pattern = r'```json(.*?)```'
                    match = re.search(json_pattern, each, re.DOTALL)
                    if match:
                        json_dict = match.group(1).strip().replace("\n", "")
                    else:
                        # Try to find JSON without code block markers
                        json_pattern = r'\{.*?"func".*?\}'
                        match = re.search(json_pattern, each, re.DOTALL)
                        if match:
                            json_dict = match.group(0)
                        else:
                            continue
                    
                    # Parse JSON
                    res_dict = json.loads(json_dict)
                    
                    # Clean and validate function
                    func = res_dict['func'].strip()
                    func = '\n'.join([line for line in func.split('\n') 
                                     if 'download' not in line and 'requests' not in line])
                    
                    try:
                        exec(func)
                    except Exception as e:
                        logger.debug(f"Error executing eval function: {e}")
                        continue
                    
                    eval_funcs.append(func)
                    
                    # Extract test cases
                    for each_case in res_dict.get('cases', []):
                        try:
                            input_val = each_case.get('input', '')
                            output_val = each_case.get('output', False)
                            
                            # Convert string 'true'/'false' to bool if needed
                            if isinstance(output_val, str):
                                output_val = output_val.lower() == 'true'
                            
                            test_cases.append((input_val, output_val))
                        except Exception as e:
                            logger.debug(f"Error processing test case: {e}")
                
                except Exception as e:
                    logger.debug(f"Error processing eval function result: {e}")
            
            # Remove duplicates
            eval_funcs = list(set(eval_funcs))
            test_cases = list(map(json.loads, set(map(json.dumps, test_cases))))
            
            # Skip if not enough functions or test cases
            # if len(eval_funcs) < 1 or len(test_cases) < 3:
            #     continue
            
            # Filter test cases
            filtered_test_cases = []
            for test_case in test_cases:
                flag = False
                for func in eval_funcs:
                    local_vars = {}
                    try:
                        exec(func, globals(), local_vars)
                    except Exception as e:
                        logger.debug(f"Error executing eval function: {e}")
                        continue
                    
                    if 'evaluate' not in local_vars:
                        continue
                    
                    eval_func = local_vars['evaluate']
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res_val = eval_func(test_case[0])
                    except Exception as e:
                        logger.debug(f"Error executing eval function: {e}")
                        res_val = None
                    finally:
                        signal.alarm(0)
                    
                    if res_val is not None and res_val == test_case[1]:
                        flag = True
                        break
                
                if flag:
                    filtered_test_cases.append(test_case)
            
            # Score functions
            scored_funcs = []
            for func in eval_funcs:
                local_vars = {}
                try:
                    exec(func, globals(), local_vars)
                except Exception as e:
                    logger.debug(f"Error executing eval function: {e}")
                    continue
                
                if 'evaluate' not in local_vars:
                    continue
                
                eval_func = local_vars['evaluate']
                acc = []
                
                for inp, out in filtered_test_cases:
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res_val = eval_func(inp)
                    except Exception as e:
                        logger.debug(f"Error executing eval function: {e}")
                        res_val = None
                    finally:
                        signal.alarm(0)
                    
                    if res_val is None or res_val != out:
                        acc.append(0)
                    else:
                        acc.append(1)
                
                acc_mean = np.mean(acc) if acc else 0
                scored_funcs.append((func, acc_mean))
            
            # Filter valid functions
            valid_funcs = [each for each in scored_funcs if each[1] >= 0.5]
            if not valid_funcs:
                continue
            
            # Add to filtered results
            filtered_generated_eval_functions.append({
                "instruction": result['instruction'],
                "eval_func": valid_funcs,
                "cases": filtered_test_cases
            })
        
        logger.info(f"Filtered to {len(filtered_generated_eval_functions)} valid evaluation functions")
        return filtered_generated_eval_functions
    
    def generate_instruction(self, filtered_generated_eval_functions, k=2, batch_size=8, show_progress=True):
        """Generate instructions from evaluation functions."""
        generated_instructions = []
        count = 0
        filter_count = 0
        
        # Build all prompts first
        all_prompts = []
        for line in filtered_generated_eval_functions:
            funcs = line["eval_func"][:3]
            instruction_prompt = f"""You are an expert in converting the Python eval function code into the corresponding instruction text. I will provide the eval function code. Please strictly follow the code to convert it into the corresponding instruction text. Here's an example: 

[["def evaluate(response):\n    return 'e' not in response.lower()", 1.0], ["def evaluate(response):\n    words = response.split()\n    for word in response.split():\n        if 'e' in word.lower():\n            return False\n    return True", 1.0], ["def evaluate(response):\n    return all('e' not in word.lower() for word in response.split())", 1.0]] 

["Answer without using any words that contain the letter 'E'.","Answer with words that do not contain the letter 'E'.","Answer with words that do not contain the letter 'E'."] Please convert the following eval function into instructions stored in a list: 

{funcs}"""
            
            for _ in range(k):
                all_prompts.append(instruction_prompt)
        
        # Process prompts in batches
        logger.info(f"Generating instructions from {len(filtered_generated_eval_functions)} eval functions with k={k}")
        all_responses = self.batch_call_llm(all_prompts, max_new_tokens=1024, batch_size=batch_size, show_progress=show_progress)
        
        # Process responses
        idx = 0
        for line in filtered_generated_eval_functions:
            back_instruction = None
            
            for _ in range(k):
                if idx < len(all_responses):
                    response = all_responses[idx]
                    idx += 1
                    
                    try:
                        # Try to parse the response as JSON
                        # First look for JSON in code blocks
                        json_pattern = r'```(json)?(.*?)```'
                        match = re.search(json_pattern, response, re.DOTALL)
                        if match:
                            json_text = match.group(2).strip()
                        else:
                            # Try to find a list directly
                            list_pattern = r'\[(.*?)\]'
                            match = re.search(list_pattern, response, re.DOTALL)
                            if match:
                                json_text = f"[{match.group(1)}]"
                            else:
                                continue
                        
                        back_instruction = json.loads(json_text)
                        break
                    except Exception:
                        filter_count += 1
                        continue
            
            if back_instruction:
                line_copy = line.copy()
                line_copy["back_instruction"] = back_instruction
                generated_instructions.append(line_copy)
                count += 1
            else:
                logger.debug(f"Failed to generate back instruction for: {line['instruction']}")
        
        logger.info(f"Generated {len(generated_instructions)} instructions, {filter_count} filtered out")
        return generated_instructions
    
    def filter_generated_instruction(self, generated_instructions, batch_size=8, show_progress=True):
        """Filter generated instructions using NLI model."""
        filtered_generated_instructions = []
        count = 0
        filter_count = 0
        
        iterator = generated_instructions
        if show_progress:
            iterator = tqdm(iterator, desc="Filtering instructions")
        
        for line in iterator:
            try:
                back_instructions = line["back_instruction"]
                ori_ins = line["instruction"]
                
                # Process each back instruction with NLI model
                nli_scores = []
                for back_ins in back_instructions[:3]:
                    premise = ori_ins
                    hypothesis = back_ins
                    
                    inputs = self.tokenizer_nli(premise, hypothesis, truncation=True, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        output = self.model_nli(**inputs)
                    prediction = torch.softmax(output.logits[0], -1).tolist()
                    label_names = ["entailment", "neutral", "contradiction"]
                    max_label = label_names[prediction.index(max(prediction))]
                    nli_scores.append(max_label)
                
                line["nli_scores"] = nli_scores
                
                # Filter out contradictions
                if "contradiction" in nli_scores:
                    filter_count += 1
                    continue
                else:
                    filtered_generated_instructions.append(line)
                    count += 1
            
            except Exception as e:
                logger.error(f"Error filtering instruction: {str(e)}")
                filter_count += 1
        
        logger.info(f"Filtered to {len(filtered_generated_instructions)} instructions, {filter_count} filtered out")
        return filtered_generated_instructions
    
    def generate_response(self, query, r1_generated_text, filtered_generated_instructions, k=2, batch_size=8, show_progress=True):
        """Generate responses following instructions."""
        generated_responses = []
        
        # Build all prompts
        all_prompts = []
        for instruction in filtered_generated_instructions:
            ins_text = instruction['instruction']
            for _ in range(k):
                prompt = (
                    f"{r1_generated_text}\n"
                    f"Re-write the above text following: {ins_text}\n\n"
                    f"Note: Use the same words and sentences but re-arrange them in a way that strictly follows the instruction.\n"
                )
                all_prompts.append(prompt)
        
        # Process prompts in batches
        if show_progress:
            logger.info(f"Generating responses for {len(filtered_generated_instructions)} instructions with k={k}")
        all_generated_texts = self.batch_call_llm(all_prompts, max_new_tokens=1024, batch_size=batch_size, show_progress=show_progress)
        
        # Process responses
        idx = 0
        for instruction in filtered_generated_instructions:
            prompt = (
                f"Please answer the query strictly following the instruction.\n"
                f"[instruction] {instruction['instruction']}\n"
                f"[Query] {query}"
            )
            
            responses = []
            for _ in range(k):
                if idx < len(all_generated_texts):
                    responses.append(all_generated_texts[idx])
                    idx += 1
            
            generated_responses.append({
                "instruction": instruction['instruction'],
                "prompt": prompt,
                "gpt-answer": responses,
                "eval_func": instruction["eval_func"],
            })
        
        if show_progress:
            logger.info(f"Generated {len(generated_responses)} response sets")
        return generated_responses
    
    def filter_generated_response(self, generated_responses, show_progress=True):
        """Filter generated responses using evaluation functions."""
        filtered_samples = []
        
        iterator = generated_responses
        if show_progress:
            iterator = tqdm(iterator, desc="Filtering responses")
        
        for result in iterator:
            # Extract evaluation functions
            eval_funcs = []
            for func, score in result['eval_func']:
                local_vars = {}
                try:
                    exec(func, globals(), local_vars)
                except Exception as e:
                    logger.debug(f"Error executing eval function: {e}")
                    continue
                
                if 'evaluate' in local_vars:
                    eval_funcs.append(local_vars['evaluate'])
            
            # Skip if no valid evaluation functions
            if not eval_funcs:
                continue
            
            # Filter responses
            filter_responses = []
            for response in result['gpt-answer']:
                acc = []
                for eval_func in eval_funcs:
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res = eval_func(response)
                    except Exception as e:
                        logger.debug(f"Error executing eval function: {e}")
                        res = None
                    finally:
                        signal.alarm(0)
                    
                    if res is not None:
                        try:
                            acc.append(int(res))
                        except Exception:
                            continue
                
                acc_mean = np.mean(acc) if acc else 0
                if acc_mean > 0:
                    filter_responses.append(response)
                    preference_item = {
                        "messages": [
                            {"role": "user", "content": result["prompt"]},
                            {"role": "assistant", "content": item.get('response', '')}
                        ],
                        "is_positive": 1
                    }
                else:
                    preference_item = {
                        "messages": [
                            {"role": "user", "content": result["prompt"]},
                            {"role": "assistant", "content": item.get('response', '')}
                        ],
                        "is_positive": 0
                    }
                    
                self.preference_data.append(preference_item)
            
            # Extract query and add filtered responses
            for each in filter_responses:
                try:
                    query_match = re.findall(r'\[Query\](.*)$', result['prompt'], re.DOTALL)
                    query = query_match[0].strip() if query_match else ""
                    
                    filtered_samples.append({
                        'instruction': result['instruction'],
                        'query': query,
                        'response': each,
                        'prompt': result['prompt'],
                        'eval_func': result['eval_func'],
                    })
                except IndexError:
                    logger.debug(f"Prompt extraction error: {result['prompt']}")
        
        # Remove duplicates
        unique_samples = []
        seen = set()
        for item in filtered_samples:
            item_json = json.dumps(item, sort_keys=True)
            if item_json not in seen:
                seen.add(item_json)
                unique_samples.append(item)
        
        if show_progress:
            logger.info(f"Filtered to {len(unique_samples)} unique responses")
        return unique_samples
        
    def filter2_generated_response(self, filtered_generated_responses, k=2, batch_size=8, show_progress=True):
        """Further filter responses based on quality scoring."""
        filtered2_generated_responses = []
        
        if not filtered_generated_responses:
            logger.warning("No responses to filter in filter2_generated_response")
            return filtered2_generated_responses
        
        prompt_template = (
            "You are an expert that is good at judging whether a response is following the instruction and query.\n"
            "[Instruction] {instruction}\n"
            "[Query] {query}\n"
            "[Response] {response}\n"
            "Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.\n"
            "You need to judge whether the response answers the query. Please first provide a detailed analysis and then give a score ranking from 0 to 10 at the last line.\n"
            "Scoring 0 means the response is totally unrelated to the query, while scoring 10 means the response is helpful and highly related to the query.\n"
            "Please only provide a score in the format `Score: {{score}}` without any other contents at the last line."
        )
        
        all_prompts = []
        for idx, each in enumerate(filtered_generated_responses):
            for k_idx in range(k):
                prompt = prompt_template.format(
                    instruction=each.get('instruction', ''),
                    query=each.get('query', ''),
                    response=each.get('response', '')
                )
                all_prompts.append((idx, prompt))
        
        if show_progress:
            logger.info(f"Quality scoring {len(filtered_generated_responses)} responses with k={k}")
        
        batch_results = {}
        for i in range(0, len(all_prompts), batch_size):
            batch_items = all_prompts[i:i+batch_size]
            batch_prompts = [item[1] for item in batch_items]
            
            try:
                batch_responses = self.batch_call_llm(
                    batch_prompts, max_new_tokens=1024, batch_size=batch_size, show_progress=False
                )
                
                for j, response in enumerate(batch_responses):
                    if i + j < len(all_prompts):
                        idx = all_prompts[i + j][0]
                        if idx not in batch_results:
                            batch_results[idx] = []
                        batch_results[idx].append(response)
            
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                
        for idx, responses in batch_results.items():
            if idx < len(filtered_generated_responses):
                item = filtered_generated_responses[idx]
                item_copy = item.copy()
                item_copy['gen'] = responses
                
                scores = []
                for gen in responses:
                    score_match = re.search(r'Score:\s*(\d+)', gen)
                    if score_match:
                        try:
                            score = int(score_match.group(1))
                            scores.append(score)
                        except ValueError:
                            continue
                
                score = np.mean(scores) if scores else 0
                if score > 5: 
                    filtered2_generated_responses.append(item_copy)
                    preference_item = {
                        "messages": [
                            {"role": "user", "content": item_copy["prompt"]},
                            {"role": "assistant", "content": item_copy['response']}
                        ],
                        "is_positive": 1
                    }
                else:
                    preference_item = {
                        "messages": [
                            {"role": "user", "content": item_copy["prompt"]},
                            {"role": "assistant", "content": item_copy['response']}
                        ],
                        "is_positive": 0
                    }
                self.preference_data.append(preference_item)
                
        if show_progress:
            logger.info(f"Final quality filter: {len(filtered2_generated_responses)} responses")
        
        return filtered2_generated_responses

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
    parser.add_argument('--max_samples', type=int, default=10, help='Maximum number of samples to process (0 for all)')
    parser.add_argument('--output_file', type=str, default="autoif_generations.jsonl", help='Output file path')
    parser.add_argument('--endpoint_url', type=str, default="http://localhost:8000/v1")
    
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
    results = autoif.compile(
        subset, 
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Save results
    with open(args.output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Results saved to {args.output_file}")