## Host via
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
## Run via 
# python eval_if_and_r.py --endpoint_url "http://localhost:8000/v1" --inference_backend vllm --model_name "Qwen/Qwen2.5-1.5B-Instruct" --data_path /share/u/harshraj/CotIF/data/cotroller_dataset-mix-wo_translation-v5-test.json --batch_size 12 --max_new_tokens 15000
# or for hf 
# python eval_if_and_r.py --inference_backend hf --model_name "Qwen/Qwen2.5-1.5B-Instruct" --data_path /share/u/harshraj/CotIF/data/cotroller_dataset-mix-wo_translation-v5-test.json --batch_size 12 --max_new_tokens 15000

import argparse
import json
import re
from loguru import logger
from collections import defaultdict
from typing import Dict, List
from functools import lru_cache
import os
import requests
import time

import openai 

from transformers import AutoTokenizer, AutoModelForCausalLM

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
    eval_replace_words_with_emojis,
    eval_partial_soln,
    eval_change_of_thought
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
    "partial_solution": eval_partial_soln,
    "change_of_thought": eval_change_of_thought,
}

def load_jsonl(dataset_path: str) -> List[Dict]:
    """Load dataset from a JSONL file."""
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    return dataset

@lru_cache(maxsize=1000)
def extract_reasoning(text):
    """Extract reasoning from within <think> tags with caching for efficiency."""
    # match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    reasoning = text.split("</think>", 1)[0]
    if reasoning:
        # reasoning = match.group(1).strip()
        if "let me recall" in reasoning.lower()[:220] or "adhere to these instructions" in reasoning.lower()[:220]:
            paragraphs = reasoning.split("\n\n")
            if len(paragraphs) > 1:
                reasoning = "\n\n".join(paragraphs[1:])
        ret = reasoning
    else:
        ret = ""
    return ret

def eval_correctness(response, gt_response):
    answer = response.rsplit("boxed{", 1)[-1].split("}", 1)[0]
    gt_answer = gt_response.rsplit("boxed{", 1)[-1].split("}", 1)[0]
    if answer.lower() == gt_answer.lower():
        return 1.0
    else:
        return 0.0

def evaluate_batch(batch_data, client, max_new_tokens=1024):
    """Process and evaluate a batch of examples."""
    batch_msgs = []
    function_names = []
    gt_responses = []
    gt_reasonings = []
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
                
            gt_reasoning = extract_reasoning(conversation[1]["content"]) if len(conversation) > 1 else ""
            
            if not gt_reasoning:
                continue
                
            batch_msgs.append([{"role": "user", "content": conversation[0]["content"]}])
            function_names.append(function_name)
            gt_responses.append(conversation[1]["content"])
            gt_reasonings.append(gt_reasoning)
            data_points.append(data_point)
    
    if not batch_msgs:
        return []
    
    # Second pass: generate all responses in one batch
    start_time = time.time()
    if client.type == "vllm":
        batch_responses = []
        for msg in batch_msgs:
            # logger.info(f"Generating response for: {msg}")
            response = client.chat.completions.create(
                model=client.model_name,
                messages=msg,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            batch_responses.append(response.choices[0].message.content)
    else:
        batch_tokenized_prompts = client.tokenizer.apply_chat_template(batch_msgs, tokenize=False)
        encoded_inputs = client.tokenizer(batch_tokenized_prompts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_inputs.input_ids.to(client.model.device)
        attention_mask = encoded_inputs.attention_mask.to(client.model.device)
        
        outputs = client.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            pad_token_id=client.tokenizer.pad_token_id,
        )
        
        batch_responses = [
            client.tokenizer.decode(output[len(input_ids[i]):], skip_special_tokens=True)
            for i, output in enumerate(outputs)
        ]
        
    generation_time = time.time() - start_time
    logger.info(f"Generated {len(batch_responses)} responses in {generation_time:.2f}s ({len(batch_responses)/generation_time:.2f} examples/s)")
    
    # Third pass: evaluate all responses sequentially
    results = []
    
    for i, (response, function_name, gt_response, gt_reasoning) in enumerate(zip(batch_responses, function_names, gt_responses, gt_reasonings)):
        try:
            extracted_reasoning = extract_reasoning(response)
            eval_function = EVAL_FUNCTION_MAP[function_name]
            
            # Evaluate directly (for instruction following)
            if not extracted_reasoning.strip():
                if_score = 0
            else:
                if_score = eval_function(extracted_reasoning, gt_reasoning)
                
            # Evaluate directly (for answer correctness)
            if not extracted_reasoning.strip():
                score = 0
            else:
                score = eval_correctness(response, gt_response)
            
            # Prepare result entry
            results.append({
                "instruction": batch_msgs[i][0]["content"],
                "function_name": function_name,
                "model_response": response,
                "model_reasoning": extracted_reasoning,
                "ground_truth-reasoning": gt_reasonings[i],
                "ground_truth-response": gt_responses[i],
                "if_score": if_score,
                "score": score,
            })
            
        except Exception as e:
            logger.error(f"Error evaluating example {i}: {e}")
    
    return results

def evaluate_responses(client, dataset, batch_size=16, save_interval=50, max_new_tokens=1024):
    """Evaluate responses with batching and periodic saving."""
    results = {
        "overall_accuracy": 0.0,
        "overall_if_accuracy": 0.0,
        "function_accuracies": defaultdict(list),
        "correctness_accuracies": defaultdict(list),
        "examples": []
    }
    
    # Process dataset in batches for efficiency
    batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating batches")):
        batch_results = evaluate_batch(batch, client, max_new_tokens)
        # Update results dictionary
        for result in batch_results:
            function_name = result["function_name"]
            if_score = result["if_score"]
            score = result["score"]
            
            results["function_accuracies"][function_name].append(if_score)
            results["correctness_accuracies"][function_name].append(score)
            results["examples"].append(result)
        
        # Save intermediate results periodically
        if (batch_idx + 1) % save_interval == 0 or batch_idx == len(batches) - 1:
            save_path = args.data_path.replace('data/', 'results/').replace(".json", f"-{args.model_name.replace('/', '-')}-results.json")
            
            # Calculate current metrics
            all_scores, all_if_scores = [], []
            for if_scores in results["function_accuracies"].values():
                all_if_scores.extend(if_scores)
            for scores in results["correctness_accuracies"].values():
                all_scores.extend(scores)
            
            current_results = results.copy()
            current_results["overall_accuracy"] = np.mean(all_scores) if all_scores else 0.0
            current_results["overall_if_accuracy"] = np.mean(all_if_scores) if all_if_scores else 0.0
            
            function_accuracies_dict = {}
            for function_name, if_scores in results["function_accuracies"].items():
                scores = results["correctness_accuracies"][function_name]
                function_accuracies_dict[function_name] = {
                    "if_accuracy": float(np.mean(if_scores)) if scores else 0.0,
                    "accuracy": float(np.mean(scores)) if scores else 0.0,
                    "count": len(scores)
                }
            current_results["function_accuracies"] = function_accuracies_dict
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(current_results, f, indent=2)
            
            logger.info(f"Saved intermediate results after batch {batch_idx+1}/{len(batches)}")
    
    # Calculate final metrics
    all_scores, all_if_scores = [], []
    for if_scores in results["function_accuracies"].values():
        scores = results["correctness_accuracies"][function_name]
        all_scores.extend(scores)
        all_if_scores.extend(if_scores)
    
    results["overall_accuracy"] = float(np.mean(all_scores)) if all_scores else 0.0
    results["overall_if_accuracy"] = float(np.mean(all_if_scores)) if all_if_scores else 0.0
    
    # Convert defaultdict to regular dict for JSON serialization
    function_accuracies_dict = {}
    for function_name, if_scores in results["function_accuracies"].items():
        scores = results["correctness_accuracies"][function_name]
        function_accuracies_dict[function_name] = {
            "accuracy": float(np.mean(scores)) if scores else 0.0,
            "if_accuracy": float(np.mean(if_scores)) if if_scores else 0.0,
            "count": len(scores)
        }
    results["function_accuracies"] = function_accuracies_dict
    
    return results

def main():
    """Main entry point function"""
    parser = argparse.ArgumentParser(description="Evaluate model's ability to follow constraints in reasoning chains.")
    
    parser.add_argument("--endpoint_url", default="http://localhost:8000/v1", type=str,
                        help="URL of the API endpoint (e.g., 'http://localhost:8000/v1')")
    parser.add_argument("--inference_backend", type=str, default="vllm")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model to use with the API")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset JSON file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing examples")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save results every N batches")
    parser.add_argument("--max_new_tokens", type=int, default=15000,
                        help="Maximum number of new tokens to generate for each response")
    
    global args
    args = parser.parse_args()

    client = type('Client', (), {})()
    if args.inference_backend == "hf":
        client.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        client.model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    else:
        client = openai.Client(base_url=args.endpoint_url, api_key="sk-hfkj")
    client.type = args.inference_backend
    client.model_name = args.model_name
    # Load dataset
    dataset = load_jsonl(args.data_path)
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Run evaluation
    total_start_time = time.time()
    results = evaluate_responses(
        client, 
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