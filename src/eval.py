## Run via 
# python eval.py --endpoint_url "http://localhost:8000/v1" --model_name "Qwen/Qwen2.5-1.5B-Instruct" --data_path /share/u/harshraj/CotIF/data/cotroller_dataset-mix-wo_translation-v5-test.json --batch_size 12 --max_new_tokens 15000

import argparse
import json
import re
from loguru import logger
from collections import defaultdict
from typing import Dict, List, Callable, Tuple, Any, Optional
from functools import lru_cache
import os
import requests
import time

import numpy as np
from tqdm import tqdm
from langdetect import detect, DetectorFactory, LangDetectException

# Import the original constraint functions
from synthesize.constraints import (
    eval_upper_case_forward,
    eval_lower_case_forward,
    eval_title_case_forward,
    eval_replace_and_ampersand_forward,
    eval_avoid_the_forward,
    eval_one_word_per_line_forward, 
    eval_use_multiple_spaces_forward,
    eval_highlight_logical_words_forward,
    eval_sentences_per_line_forward,
    eval_commas_to_semicolons_forward,
    eval_full_stops_to_exclamation_marks_forward,
    eval_add_line_numbers_forward,
    eval_json_of_paragraphs_forward,
    eval_bracket_sentences_forward,
    eval_indent_paragraphs_forward,
    eval_insert_sentence_divider_forward,
    eval_render_as_html_forward,
    eval_translate_forward,
    
    eval_upper_case_backward,
    eval_title_case_backward,
    eval_palindromes_backwards,
    eval_hypernyms_backward,
    eval_paragraph_count_backward,
    eval_include_word_forward,
    eval_contains_list_or_enumeration_forward,
    eval_replace_words_with_emojis
)

# Set fixed seeds for reproducibility
DetectorFactory.seed = 0
np.random.seed(42)

# Map function names to their implementations
EVAL_FUNCTION_MAP = {
    "upper_case_forward": eval_upper_case_forward,
    "lower_case_forward": eval_lower_case_forward,
    "title_case_forward": eval_title_case_forward,
    "replace_and_ampersand_forward": eval_replace_and_ampersand_forward,
    "avoid_the_forward": eval_avoid_the_forward,
    "one_word_per_line_forward": eval_one_word_per_line_forward,
    "use_multiple_spaces_forward": eval_use_multiple_spaces_forward,
    "highlight_logical_words_forward": eval_highlight_logical_words_forward,
    "sentences_per_line_forward": eval_sentences_per_line_forward,
    "commas_to_semicolons_forward": eval_commas_to_semicolons_forward,
    "full_stops_to_exclamation_marks_forward": eval_full_stops_to_exclamation_marks_forward,
    "add_line_numbers_forward": eval_add_line_numbers_forward,
    "json_of_paragraphs_forward": eval_json_of_paragraphs_forward,
    "bracket_sentences_forward": eval_bracket_sentences_forward,
    "indent_paragraphs_forward": eval_indent_paragraphs_forward,
    "insert_sentence_divider_forward": eval_insert_sentence_divider_forward,
    "render_as_html_forward": eval_render_as_html_forward,
    
    "upper_case_backward": eval_upper_case_backward,
    "title_case_backward": eval_title_case_backward,
    "palindromes_backward": eval_palindromes_backwards,
    "hypernyms_backward": eval_hypernyms_backward,
    "paragraph_count_backward": eval_paragraph_count_backward,
    "include_word_forward": eval_include_word_forward,
    "contains_list_or_enumeration_forward": eval_contains_list_or_enumeration_forward,
    "replace_words_with_emojis": eval_replace_words_with_emojis,
}

def load_jsonl(dataset_path: str) -> List[Dict]:
    """Load dataset from a JSONL file."""
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    return dataset

class APIClient:
    """A client for interacting with API endpoints for language models."""
    
    def __init__(self, endpoint_url, api_key="dummy-key", model_name=None):
        """
        Initialize the API client.
        
        Args:
            endpoint_url: URL of the API endpoint
            api_key: API key for authentication
            model_name: Name of the model to use
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_name = model_name
        
        # Detect if we're using OpenAI or vLLM-compatible API
        self.api_type = "openai" if "openai.com" in endpoint_url else "vllm"
        
        # Set up session for reusing connections
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        })
        
        logger.info(f"Initialized API client for endpoint: {endpoint_url}")
        logger.info(f"Using API type: {self.api_type}")
        logger.info(f"Model name: {model_name}")
    
    def generate_completions(self, prompts, max_tokens=1024, temperature=0.0, timeout=60):
        """
        Generate completions for multiple prompts.
        
        Args:
            prompts: List of prompt strings
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            timeout: Request timeout in seconds
            
        Returns:
            List of generated texts
        """
        results = []
        
        # OpenAI-compatible endpoint
        if self.endpoint_url.endswith("/v1"):
            url = f"{self.endpoint_url}/completions"
            
            batch_results = []
            for prompt in prompts:
                try:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                    
                    response = self.session.post(url, json=payload, timeout=timeout)
                    response.raise_for_status()
                    
                    response_json = response.json()
                    generated_text = response_json.get("choices", [{}])[0].get("text", "")
                    batch_results.append(generated_text)
                    
                except Exception as e:
                    logger.error(f"Error in API call: {str(e)}")
                    batch_results.append("")  # Empty string on error
            
            results.extend(batch_results)
        
        # OpenAI Chat API
        elif "/chat/completions" in self.endpoint_url or self.api_type == "openai":
            url = self.endpoint_url if "/chat/completions" in self.endpoint_url else f"{self.endpoint_url}/chat/completions"
            
            batch_results = []
            for prompt in prompts:
                try:
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                    
                    response = self.session.post(url, json=payload, timeout=timeout)
                    response.raise_for_status()
                    
                    response_json = response.json()
                    generated_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    batch_results.append(generated_text)
                    
                except Exception as e:
                    logger.error(f"Error in API call: {str(e)}")
                    batch_results.append("")  # Empty string on error
            
            results.extend(batch_results)
            
        # Custom API format - you can extend this for other API types
        else:
            logger.error(f"Unsupported API endpoint: {self.endpoint_url}")
            results = [""] * len(prompts)  # Empty results
        
        return results

def format_chat_messages(messages):
    """
    Format chat messages into a string prompt.
    This is a simplified version and might need to be adjusted based on the API's requirements.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Formatted prompt string
    """
    formatted_prompt = ""
    for message in messages:
        role = message.get("role", "").capitalize()
        content = message.get("content", "")
        formatted_prompt += f"{role}: {content}\n\n"
    
    return formatted_prompt.strip()

def batch_generate_responses(api_client, batch_messages, max_new_tokens=1024, batch_size=8):
    """Generate responses in batches for efficiency."""
    # Prepare prompts from chat templates
    prompts = []
    for messages in batch_messages:
        # Format as string since we don't have a tokenizer for chat templates
        prompt = format_chat_messages(messages)
        prompts.append(prompt)
    
    # Process in smaller batches to avoid rate limits
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_results = api_client.generate_completions(
            batch_prompts, 
            max_tokens=max_new_tokens,
            temperature=0.0
        )
        results.extend(batch_results)
    
    return results

@lru_cache(maxsize=1000)
def extract_reasoning(text):
    """Extract reasoning from within <think> tags with caching for efficiency."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        if "let me recall" in reasoning.lower().split()[:100] or "adhere to these instructions" in reasoning.lower().split()[:100]:
            paragraphs = reasoning.split("\n\n")
            if len(paragraphs) > 1:
                reasoning = "\n\n".join(paragraphs[1:])
        return reasoning
    return ""

def evaluate_batch(batch_data, api_client, max_new_tokens=1024):
    """Process and evaluate a batch of examples."""
    batch_msgs = []
    function_names = []
    gt_texts = []
    data_points = []
    
    # First pass: prepare all inputs
    for data_point in batch_data:
        if not data_point.get("messages"):
            continue
        
        applied_functions = data_point.get("applied_functions", [])
        applied_functions = [applied_functions] if isinstance(applied_functions, str) else applied_functions
        
        for function_name in applied_functions:
            if function_name not in EVAL_FUNCTION_MAP:
                continue
                
            conversation = data_point["messages"]
            if len(conversation) < 2:
                continue
                
            gt_text = extract_reasoning(conversation[1]["content"]) if len(conversation) > 1 else ""
            
            if not gt_text:
                continue
                
            batch_msgs.append([{"role": "user", "content": conversation[0]["content"]}])
            function_names.append(function_name)
            gt_texts.append(gt_text)
            data_points.append(data_point)
    
    if not batch_msgs:
        return []
    
    # Second pass: generate all responses in one batch
    start_time = time.time()
    batch_responses = batch_generate_responses(
        api_client, 
        batch_msgs, 
        max_new_tokens=max_new_tokens
    )
    generation_time = time.time() - start_time
    logger.info(f"Generated {len(batch_responses)} responses in {generation_time:.2f}s ({len(batch_responses)/generation_time:.2f} examples/s)")
    
    # Third pass: evaluate all responses sequentially
    results = []
    
    for i, (response, function_name, gt_text) in enumerate(zip(batch_responses, function_names, gt_texts)):
        try:
            extracted_reasoning = extract_reasoning(response)
            eval_function = EVAL_FUNCTION_MAP[function_name]
            
            # Evaluate directly
            score = eval_function(extracted_reasoning, gt_text)
            
            # Prepare result entry
            results.append({
                "instruction": batch_msgs[i][0]["content"],
                "function_name": function_name,
                "model_response": response,
                "model_reasoning": extracted_reasoning,
                "ground_truth": gt_texts[i],
                "score": score
            })
            
        except Exception as e:
            logger.error(f"Error evaluating example {i}: {e}")
    
    return results

def evaluate_responses(api_client, dataset, batch_size=16, save_interval=50, max_new_tokens=1024):
    """Evaluate responses with batching and periodic saving."""
    results = {
        "overall_accuracy": 0.0,
        "function_accuracies": defaultdict(list),
        "examples": []
    }
    
    # Process dataset in batches for efficiency
    batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating batches")):
        batch_results = evaluate_batch(batch, api_client, max_new_tokens)
        # Update results dictionary
        for result in batch_results:
            function_name = result["function_name"]
            score = result["score"]
            
            results["function_accuracies"][function_name].append(score)
            results["examples"].append(result)
        
        # Save intermediate results periodically
        if (batch_idx + 1) % save_interval == 0 or batch_idx == len(batches) - 1:
            save_path = args.data_path.replace(".json", f"-{args.model_name.replace('/', '-')}-results.json")
            
            # Calculate current metrics
            all_scores = []
            for scores in results["function_accuracies"].values():
                all_scores.extend(scores)
            
            current_results = results.copy()
            current_results["overall_accuracy"] = np.mean(all_scores) if all_scores else 0.0
            
            function_accuracies_dict = {}
            for function_name, scores in results["function_accuracies"].items():
                function_accuracies_dict[function_name] = {
                    "accuracy": float(np.mean(scores)) if scores else 0.0,
                    "count": len(scores)
                }
            current_results["function_accuracies"] = function_accuracies_dict
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(current_results, f, indent=2)
            
            logger.info(f"Saved intermediate results after batch {batch_idx+1}/{len(batches)}")
    
    # Calculate final metrics
    all_scores = []
    for scores in results["function_accuracies"].values():
        all_scores.extend(scores)
    
    results["overall_accuracy"] = float(np.mean(all_scores)) if all_scores else 0.0
    
    # Convert defaultdict to regular dict for JSON serialization
    function_accuracies_dict = {}
    for function_name, scores in results["function_accuracies"].items():
        function_accuracies_dict[function_name] = {
            "accuracy": float(np.mean(scores)) if scores else 0.0,
            "count": len(scores)
        }
    results["function_accuracies"] = function_accuracies_dict
    
    return results

def main():
    """Main entry point function"""
    parser = argparse.ArgumentParser(description="Evaluate model's ability to follow constraints in reasoning chains.")
    
    parser.add_argument("--endpoint_url", type=str, required=True,
                        help="URL of the API endpoint (e.g., 'http://localhost:8000/v1')")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model to use with the API")
    parser.add_argument("--api_key", type=str, default="dummy-key",
                        help="API key for authentication (default: 'dummy-key')")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset JSON file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing examples")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save results every N batches")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate for each response")
    
    global args
    args = parser.parse_args()
    
    # Configure detailed logging
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(
        args.data_path.replace(".json", f"-{args.model_name.replace('/', '-')}-log.txt"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )

    # Initialize API client
    api_client = APIClient(
        endpoint_url=args.endpoint_url,
        api_key=args.api_key,
        model_name=args.model_name
    )
    
    # Load dataset
    dataset = load_jsonl(args.data_path)
    logger.info(f"Loaded {len(dataset)} examples from {args.data_path}")
    
    # Run evaluation
    total_start_time = time.time()
    results = evaluate_responses(
        api_client, 
        dataset, 
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        max_new_tokens=args.max_new_tokens
    )
    total_time = time.time() - total_start_time
    
    # Print and save results
    logger.info(f"\nEvaluation completed in {total_time:.2f}s")
    logger.info(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    
    logger.info("\nAccuracy by function:")
    for function_name, function_results in sorted(
        results["function_accuracies"].items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    ):
        accuracy = function_results["accuracy"]
        count = function_results["count"]
        logger.info(f"  {function_name}: {accuracy:.4f} ({count} examples)")
    
    save_path = args.data_path.replace(".json", f"-{args.model_name.replace('/', '-')}-results.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()