## Host via
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
## Run via 
from datasets import load_dataset
# python eval_math.py --endpoint_url "http://localhost:8000/v1" --inference_backend vllm --model_name "Qwen/Qwen2.5-1.5B-Instruct" --data_path /share/u/harshraj/CotIF/data/cotroller_dataset-mix-wo_translation-v5-test.json --batch_size 12 --max_new_tokens 15000 --output_path /share/u/harshraj/CotIF/results/MATH500-R1_Distill_Llama_8B.json
# or for hf
# python eval_math.py --inference_backend hf --model_name "Qwen/Qwen2.5-1.5B-Instruct" --data_path /share/u/harshraj/CotIF/data/cotroller_dataset-mix-wo_translation-v5-test.json --batch_size 12 --max_new_tokens 15000

import re
import argparse
import time
import json
import math
import logging

import sys
sys.path.append("/disk/u/harshraj/CotIF/src")
from math500.grader import grade_answer

from datasets import load_dataset
from pylatexenc.latex2text import LatexNodes2Text
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# QUERY_TEMPLATE = """
# Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

# {QUESTION}

# Remember to put your answer in latex code on its own line after "Answer:", and you do not need to use a \\boxed command.
# """.strip()

def eval_pred(generated_text, ref):
    matches = re.search("</think>", generated_text)
    if matches is None:
        return "", generated_text, ref
    generated_answer = generated_text[matches.end():]
    # in generated answer select the content of \boxed{}
    matches = re.search("\\\\boxed{", generated_answer)
    if matches is None:
        return "", generated_text, ref
    generated_answer = generated_answer[matches.end():]
    # search all the way to the end of the string   }
    reversed = generated_answer[::-1]
    matches = re.search("}", reversed)
    if matches is None:
        return "", generated_text, ref
    generated_answer = generated_answer[:len(generated_answer) - matches.start()]
    return grade_answer(generated_answer, ref), generated_answer, ref

# converter = LatexNodes2Text()
# def eval_pred(pred, gt):
#     try:
#         ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
#         match = re.search(ANSWER_PATTERN, pred)
#         extracted_pred = match.group(1) if match else None
#         extracted_pred = converter.latex_to_text(extracted_pred)
#         extracted_pred = extracted_pred.replace(' ', '')

#         extracted_gt = converter.latex_to_text(gt)
#         extracted_gt = extracted_gt.replace(' ', '')
#         return extracted_gt==extracted_pred, extracted_pred, extracted_gt
#     except Exception as e:
#         logger.info(f"exception while scroing: {e}") 
#         return False, "error in extracting from pred", "error in extracting from gt"
    
def evaluate_responses(client,
                       dataset,
                       batch_size: int,
                       save_interval: int,
                       max_new_tokens: int):
    num_samples = len(dataset)
    num_batches = math.ceil(num_samples / batch_size)
    correct = 0
    records = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        batch = dataset.select(range(start,end))

        batch_msgs = []
        references = []
        for ex in batch:
            # print("ex:", ex)
            prompt = ex.get("problem")
            batch_msgs.append([
                {"role": "user", "content": prompt+ "Please reason step by step, and put your final answer within \\boxed{}."}
            ])
            references.append(ex.get("answer"))

        start_time = time.time()
        if client.type == "vllm":
            batch_responses = []
            for msg in batch_msgs:
                resp = client.chat.completions.create(
                    model=client.model_name,
                    messages=msg,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )
                batch_responses.append(resp.choices[0].message.content)
        else:
            tokenized = client.tokenizer.apply_chat_template(
                batch_msgs, tokenize=False
            )
            encoded = client.tokenizer(
                tokenized,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoded.input_ids.to(client.model.device)
            attention_mask = encoded.attention_mask.to(client.model.device)
            outputs = client.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=client.tokenizer.pad_token_id,
            )
            batch_responses = [
                client.tokenizer.decode(
                    out[input_ids[i].size(0):], skip_special_tokens=True
                )
                for i, out in enumerate(outputs)
            ]
        gen_time = time.time() - start_time
        logger.info(
            f"Batch {batch_idx+1}/{num_batches}: generated "
            f"{len(batch_responses)} in {gen_time:.2f}s "
            f"({len(batch_responses)/gen_time:.2f} ex/s)"
        )

        for prompt, pred, ref in zip(
            [m[-1]["content"] for m in batch_msgs],
            batch_responses,
            references
        ):
            is_corr, extracted_pred, extracted_gt = eval_pred(pred,ref)
            if not isinstance(is_corr, str):
                correct += is_corr
                records.append({
                    "prompt": prompt,
                    "prediction": pred,
                    "reference": ref,
                    "correct": is_corr,
                    "extracted_gt":extracted_gt,
                    "extracted_pred": extracted_pred
                })
        if save_interval > 0 and (batch_idx + 1) % save_interval == 0:
            partial_path = args.output_path
            interim = {
                "overall_accuracy": correct / len(records),
                "results": records
            }
            with open(partial_path, "w") as fp:
                json.dump(interim, fp, indent=2)
            logger.info(f"Saved partial results to {partial_path}")

    overall_accuracy = correct / num_samples
    return {"overall_accuracy": overall_accuracy, "results": records}

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model's constraint-following on reasoning chains"
    )
    parser.add_argument(
        "--endpoint_url", type=str, default=None,
        help="Custom API base URL (for openai backend)"
    )
    parser.add_argument(
        "--api_key", type=str, default="sk-hfnv",
        help="API key (for openai backend)"
    )
    parser.add_argument(
        "--inference_backend", type=str, choices=["hf", "vllm", "openai"],
        default="hf", help="Which backend to use"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Model name or path"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="HF dataset identifier or local path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Examples per batch"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1,
        help="Save partial results every N batches"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=8000,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--output_path", type=str,
    )
    args_ = parser.parse_args()
    global args
    args = args_

    if args.inference_backend == "hf":
        client = type("Client", (), {})()
        client.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        client.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map="auto"
        )
    elif args.inference_backend=="vllm":
        client = openai.Client(
            base_url=args.endpoint_url, api_key=args.api_key
        )

    client.type = args.inference_backend
    client.model_name = args.model_name

    dataset = load_dataset(args.data_path, split="test")
    logger.info(f"Loaded dataset `{args.data_path}` with {len(dataset)} samples")

    tic = time.time()
    results = evaluate_responses(
        client,
        dataset,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        max_new_tokens=args.max_new_tokens
    )
    elapsed = time.time() - tic

    logger.info(f"Evaluation done in {elapsed:.2f}s")
    logger.info(f"Overall accuracy: {results['overall_accuracy']:.4f}")

    save_path = args.output_path
    with open(save_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info(f"Results written to {save_path}")

if __name__ == "__main__":
    main()
