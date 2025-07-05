# run using
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 8 --host 0.0.0.0 --port 8000
# python -m hehe_partial_sol_datasyn

import os 
import re
import random
import json
import time
import asyncio
from tqdm.auto import tqdm
from typing import List

import openai
from openai import OpenAIError
from datasets import load_dataset


NEW_PATH_KEYWORDS = [
    "wait", "double-check", "alternatively",
"make sure","another way","verify", "to confirm"
]

def extract_think_content(response: str) -> str:
    """
    Extracts all content enclosed in <think>...</think> tags and joins them.
    """
    response = response.split("<think>", 1)[-1]
    return response.rsplit("</think>", 1)[0]

def split_into_paragraphs(text: str) -> list[str]:
    """
    Splits the reasoning trace into paragraphs using double-newline delimiters.
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_paragraphs(paragraphs: list[str], keywords: list[str]) -> list[str]:
    """
    Groups paragraphs into chunks. A new chunk starts whenever a paragraph
    contains any of the specified keywords (case-insensitive).
    """
    chunks = []
    current_chunk = []
    for para in paragraphs:
        if any(kw.lower() in para.lower() for kw in keywords) and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
        else:
            current_chunk.append(para)
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

async def query_chat(prompt, model="gpt-4o-mini"):
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
        timeout=100, 
    )
    return response.choices[0].message.content

async def annotate_chunks(batch, model: str = "gpt-4o-mini") -> dict:
#     prompt = f"""Given a reasoning trace, along with a ground-truth answer, evaluate it. If the trace reaches a result at the end, return the intermediate result; otherwise,return None if the chunk does not contain an intermediate result (e.g., pure reflections). 
#     Then, if an intermediate answer exists, compare it to the ground-truth answer. If the intermediate result in the chunk equals the ground-truth answer, return True; if the intermediate result in the chunk does not equal the ground-truth answer, return False; if no intermediate answer exists, return None.
# Output in JSON format:

# ```json
# {{"id": "1", "result": "6 + 9i" / null, "correct": true / false / null}},
# ```

# Input chunk: {chunk}
# Ground-truth answer: {answer}
# """
    prompts = []
    for item in batch:
        prompt = f"""Given several chunks of a reasoning trace, along with a ground-truth answer, independently evaluate each chunk. If a chunk reaches a result at the end, return the intermediate result; otherwise, return null if the chunk does not contain an intermediate result (e.g., pure reflections). 
Then, if an intermediate answer exists, compare it to the ground-truth answer. If the intermediate result in the chunk **leads** to the ground-truth answer, return true; if the intermediate result in the chunk **does not lead** to the ground-truth answer, return false; if no intermediate answer exists, return null.
Output in JSON format:

```json
[
{{"id": "1", "result": "6 + 9i" / null, "correct": true / false / null}},
...
]
```

Input chunks: {item['chunks']}
Ground-truth answer: {item['answer']}
"""   
#         prompt = f"""You are given several chunks from a reasoning trace produced by a model. Your task is to analyze whether a chunk represents a **change of direction**, where the model revises, contradicts, or abandons an earlier assumption, plan, or line of reasoning and adopts a new one.

# For each chunk, answer the following:
# 1. Does this chunk **revise** or **contradict** an earlier hypothesis or approach? 
# 2. If yes, briefly describe the **nature of the change** (e.g., "revises earlier assumption about value of x", or "abandons algebraic approach in favor of numeric estimation").
# 3. If not, mark it as a continuation or elaboration of the prior reasoning.

# Return results in the following JSON format:

# ```json
# [
#   {{
#     "id": "1",
#     "shift": true / false,
#     "description": "..."  // Only if shift is true, otherwise null
#   }},
#   ...
# ]
# ```

# # Input chunks: {item['chunks']}
# # Ground-truth answer: {item['answer']}
# """
        prompts.append(prompt)
    max_retries, backoff_factor = 12, 8
    for attempt in range(1, max_retries + 1):
        try:
            # responses = openai_client.chat.completions.create(
            #     model=model,
            #     messages=[[
            #         {"role": "user", "content": prompt}
            #     ] for prompt in prompts],
            #     temperature=0.1
            # )
            tasks = [query_chat(prompt, model) for prompt in prompts]
            responses = await asyncio.gather(*tasks)
        except OpenAIError as e:
            print(f"[Attempt {attempt}] OpenAI API error: {e}")
            if attempt == max_retries:
                raise  
            time.sleep(backoff_factor ** attempt) 
          
    results = []
    for response, item in zip(responses, batch):  
        raw = response.strip()
        raw = raw.split("```json", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            result = json.loads(raw)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating response: {e}")
            results.append([{"id": 1, "result": item['answer'], "correct": True}])
    return results

def merge_annotated_chunks(annotated: list[dict]) -> list[dict]:
    """
    Merge chunks without answers into the nearest chunk that has an answer.
    Returns a list of merged chunk dicts, each with 'chunk', 'answer', and 'correct'.
    """
    # Identify indices of chunks with answers
    answer_indices = [i for i, item in enumerate(annotated) if item['answer'] is not None]
    groups: dict[int, list[int]] = {i: [i] for i in answer_indices}

    # Assign no-answer chunks to nearest answer chunk
    for i, item in enumerate(annotated):
        if item['answer'] is None:
            if not answer_indices:
                continue
            nearest = min(answer_indices, key=lambda ai: abs(ai - i))
            groups[nearest].append(i)

    # Build merged chunks
    merged = []
    for ans_idx, idxs in groups.items():
        merged_text = "\n\n".join(annotated[i]['chunk'] for i in sorted(idxs))
        merged.append({
            'chunk': merged_text,
            'answer': annotated[ans_idx]['answer'],
            'correct': annotated[ans_idx]['correct']
        })
    return merged

def generate_response(question):
    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        messages=[{"role": "user", "content": f"{question}\nPlease reason step by step, and put your final answer within \boxed{{}}."}],
        max_tokens=16000,
        temperature=0.0,
    )
    return resp.choices[0].message.content

def format_assistant_response(assistant):
    assistant = assistant.replace('<|begin_of_thought|>', '<think>')
    assistant = assistant.replace('<|end_of_thought|>', '</think>')
    assistant = assistant.replace('<|begin_of_solution|>', '')
    assistant = assistant.replace('<|end_of_solution|>', '')
    assistant = re.sub(r'<think>\n+', '<think>\n', assistant)
    assistant = re.sub(r'\n+</think>', '\n</think>', assistant)
    assistant = re.sub(r'</think>\n+', '</think>\n\n', assistant)
    assistant = re.sub(r'\n*$', '', assistant)
    return assistant

def parse_answer(generated_text):
    matches = re.search("</think>", generated_text)
    if matches is None:
        return ""
    generated_answer = generated_text[matches.end():]
    # in generated answer select the content of \boxed{}
    matches = re.search("\\\\boxed{", generated_answer)
    if matches is None:
        return ""
    generated_answer = generated_answer[matches.end():]
    # search all the way to the end of the string   }
    reversed = generated_answer[::-1]
    matches = re.search("}", reversed)
    if matches is None:
        return ""
    generated_answer = generated_answer[:len(generated_answer) - matches.start()]
    return generated_answer

import argparse
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--method", type=str, required=True)
args = parser.parse_args()

api_key = os.getenv("OPENAI_API_KEY", "sk-")
openai_client = openai.AsyncClient(base_url="http://localhost:8000/v1", api_key=api_key) # base_url="http://localhost:8000/v1", 
client = openai.Client(base_url="http://localhost:8001/v1", api_key="sk-cdnnc")

batch_size = 16
DATA_PATH = "bespokelabs/Bespoke-Stratos-17k" # or "HuggingFaceH4/MATH-500" 
METHOD = args.method # or "from_base" or "from_ft"
if METHOD == "from_base":
    ds = load_dataset(DATA_PATH, split="train")
    # ds = ds.select(range(5000, 12000))
    ds = ds.select(range(5000)) # when from_base
else:
    METHOD = METHOD+'-s2.1'
    with open("/disk/u/harshraj/CotIF/data/Bespoke-Stratos-17k-llama3-hehe-partial_soln-from_ft.jsonl", 'r') as file:
        ds = [json.loads(line) for line in file]
        ds = [{'conversations': [{"role": "user", "value": datum['messages'][0]['content']},
                                 {"role": "assistant", "value": datum['messages'][1]['content']}]} for datum in ds]

async def main():
    annotateds, batch_input = [], []
    for datum in tqdm(ds, desc="Processing dataset"):
        if "bespoke" in DATA_PATH:
            if len(datum['conversations'])!=2:
                continue
            question = datum['conversations'][0]['value']
            raw_answer = datum['conversations'][1]['value']
            raw_answer = format_assistant_response(raw_answer)
            answer = parse_answer(raw_answer)
        elif "MATH-500" in DATA_PATH:
            question, answer = datum['problem'], datum['answer']
            raw_answer = generate_response(question)
        trace = extract_think_content(raw_answer)
        paragraphs = split_into_paragraphs(trace)
        chunks = chunk_paragraphs(paragraphs, NEW_PATH_KEYWORDS)
        batch_input.append({"question": question, "raw_answer": raw_answer,
                            "trace": trace, "answer": answer,
                            "chunks":chunks})
        
    messages = []
    for i in tqdm(range(0, len(batch_input), batch_size), desc="Processing batches"):
        batch = batch_input[i:i + batch_size]
        results = await annotate_chunks(batch, "meta-llama/Llama-3.3-70B-Instruct")
        annotated_batch = [{"question": item["question"],
                            "raw_answer": item["raw_answer"],
                            "trace": item["trace"],
                            "final_answer": item["answer"],
                            "partial_solutions-annotation": result,
                            "partial_solutions-chunks": item["chunks"],} for item, result in zip(batch, results)]
        annotateds.extend(annotated_batch)
        
        for datum in annotated_batch:
            if len(datum["partial_solutions-annotation"])<=1:
                continue
            message = {"messages":
                [{"role": "user", "content": datum["question"]},
                {"role": "assistant", "content": ""}],
            "applied_functions": "partial_solution"}
            annotated_thought = ""
            for soln, chunk in zip(
                datum["partial_solutions-annotation"],
                datum["partial_solutions-chunks"]
            ):
                if "result" not in soln:
                    continue
                if soln['result'] is not None:
                    soln['result'] = str(soln['result'])
                    pos = chunk.find(soln['result'])
                    if pos != -1:
                        start = chunk[:pos],
                        end   = chunk[pos + len(str(soln['result'])):]
                        annotated_chunk = f"{start}<partial_solution>{soln['result']}</partial_solution>{end}"
                    else:
                        annotated_chunk = chunk
                else:
                    annotated_chunk = chunk
                annotated_thought += "\n" + annotated_chunk
            if "<partial_solution>" not in annotated_thought:
                continue
            after_think = datum['raw_answer'].split('</think>')[-1]
            message['messages'][1]["content"] = f"{annotated_thought.strip('()')}</think>{after_think}"
            messages.append(message)
        # break
        train_len1, train_len2 = int(0.9*len(annotateds)), int(0.9*len(messages))
        with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-meta-{METHOD}.jsonl", "w") as file:
            for item in annotateds[:train_len1]:
                file.write(json.dumps(item) + "\n")
        with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-{METHOD}.jsonl", "w") as file:
            for item in messages[:train_len2]:
                file.write(json.dumps(item) + "\n")
                
        with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-meta-test-{METHOD}.jsonl", "w") as file:
            for item in annotateds[train_len1:]:
                file.write(json.dumps(item) + "\n")
        with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-test-{METHOD}.jsonl", "w") as file:
            for item in messages[train_len2:]:
                file.write(json.dumps(item) + "\n")
                    
if __name__ == "__main__":
    asyncio.run(main())
    description = random.choice([
        'Annotate the intermediate steps you produce during reasoning using <partial_solution>...</partial_solution> tags',
        'Wrap the intermediate solutions in your reasoning chain with <partial_solution>solution</partial_solution> tags',
        'For every intermediate answer in your reasoning, annotate it as <partial_solution>solution</partial_solution>',
        'Mark intermediate results in your chain-of-thought using <partial_solution>solution</partial_solution> tags',
        'Enclose each intermediate solution you derive within <partial_solution>solution</partial_solution> tags',
        'As you reason step-by-step, annotate partial answers with <partial_solution>solution</partial_solution>',
        'Tag any intermediate solution you get with <partial_solution>solution</partial_solution> to highlight it',
        'Whenever you find a partial solution while reasoning, annotate it as <partial_solution>solution</partial_solution>',
        'Each time you reach an intermediate conclusion, wrap it in <partial_solution>solution</partial_solution> tags',
        'Use <partial_solution>solution</partial_solution> to mark any partial answers that emerge in your reasoning process'
    ])
    with open(f"/disk/u/harshraj/CotIF/data/Bespoke-Stratos17k-llama3-hehe-partial_soln-{METHOD}.jsonl", "r") as file:
        data = [json.loads(line) for line in file]
    for datum in data:
        datum['messages'][0]['content']=datum['messages'][0]['content'].split("\n\n## NOTES\n\nInstructions that you must follow in your reasoning chain (the part of your", 1)[0] + f"\n\n## NOTES\n\nInstructions that you must follow in your reasoning chain (the part of your response before your final answer):\n- {description}"
        if datum['messages'][1]['content'].count("Okay, so before I start reasoning, le") > 1:
            datum['messages'][1]['content']=f"<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: I must {description}.\nIt's really important that I adhere to these instructions in my reasoning chain, since otherwise the user's need would not be met. So from the next paragraph onward, I'll {description}.\n\n"+datum['messages'][1]['content'].rsplit("\n\n<think>\n", 1)[-1]
        
    with open(f"/disk/u/harshraj/CotIF/data/Bespoke-Stratos17k-llama3-hehe-partial_soln-{METHOD}.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        
    description = random.choice([
        'Annotate the intermediate steps you produce during reasoning using <partial_solution>...</partial_solution> tags',
        'Wrap the intermediate solutions in your reasoning chain with <partial_solution>solution</partial_solution> tags',
        'For every intermediate answer in your reasoning, annotate it as <partial_solution>solution</partial_solution>',
        'Mark intermediate results in your chain-of-thought using <partial_solution>solution</partial_solution> tags',
        'Enclose each intermediate solution you derive within <partial_solution>solution</partial_solution> tags',
        'As you reason step-by-step, annotate partial answers with <partial_solution>solution</partial_solution>',
        'Tag any intermediate solution you get with <partial_solution>solution</partial_solution> to highlight it',
        'Whenever you find a partial solution while reasoning, annotate it as <partial_solution>solution</partial_solution>',
        'Each time you reach an intermediate conclusion, wrap it in <partial_solution>solution</partial_solution> tags',
        'Use <partial_solution>solution</partial_solution> to mark any partial answers that emerge in your reasoning process'
    ])    
    with open(f"/disk/u/harshraj/CotIF/data/Bespoke-Stratos17k-llama3-hehe-partial_soln-test-{METHOD}.jsonl", "r") as file:
        data = [json.loads(line) for line in file]
    for datum in data:
        datum['messages'][0]['content']=datum['messages'][0]['content'].split("\n\n## NOTES\n\nInstructions that you must follow in your reasoning chain (the part of your", 1)[0] + f"\n\n## NOTES\n\nInstructions that you must follow in your reasoning chain (the part of your response before your final answer):\n- {description}"
        if datum['messages'][1]['content'].count("Okay, so before I start reasoning, le") > 1:
            datum['messages'][1]['content']=f"<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: I must {description}.\nIt's really important that I adhere to these instructions in my reasoning chain, since otherwise the user's need would not be met. So from the next paragraph onward, I'll {description}.\n\n"+datum['messages'][1]['content'].rsplit("\n\n<think>\n", 1)[-1]
        
    with open(f"/disk/u/harshraj/CotIF/data/Bespoke-Stratos17k-llama3-hehe-partial_soln-test-{METHOD}.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")