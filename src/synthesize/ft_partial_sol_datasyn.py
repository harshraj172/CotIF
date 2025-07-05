# run using
# python -m vllm.entrypoints.openai.api_server --model /disk/u/harshraj/CotIF/models/R1-Distill-Llama-8B-SFT-cotroller_dataset-bespoke-35k_all_cotif-w_partial_soln --tensor-parallel-size 8 --host 0.0.0.0 --port 8000
# python -m ft_partial_sol_datasyn

import os 
import re
import json
import asyncio
from tqdm.auto import tqdm

import openai
from datasets import load_dataset


def extract_think_content(response: str) -> str:
    """
    Extracts all content enclosed in <think>...</think> tags and joins them.
    """
    response = response.split("<think>", 1)[-1]
    return response.rsplit("</think>", 1)[0]


async def generate_response(question, model_name):
    resp = await client.chat.completions.create(
        model=model_name,
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


api_key = os.getenv("OPENAI_API_KEY", "sk-")
client = openai.AsyncClient(base_url="http://localhost:8000/v1", api_key=api_key)

batch_size = 22
MODEL_NAME="/disk/u/harshraj/CotIF/models/R1-Distill-Llama-8B-SFT-cotroller_dataset-bespoke-35k_all_cotif-w_partial_soln"
DATA_PATH = "bespokelabs/Bespoke-Stratos-17k" # or "HuggingFaceH4/MATH-500" 
ds = load_dataset(DATA_PATH, split="train")
# ds = ds.select(range(5000, 12000))
ds = ds.select(range(5000))

async def main():
    annotateds, batch_input = [], []
    for datum in tqdm(ds, desc="Processing dataset"):
        if "bespoke" in DATA_PATH:
            if len(datum['conversations'])!=2:
                continue
            question = datum['conversations'][0]['value']
        elif "MATH-500" in DATA_PATH:
            question = datum['problem'], datum['answer']
        batch_input.append({"question": question})
        
    messages = []
    for i in tqdm(range(0, len(batch_input), batch_size), desc="Processing batches"):
        batch = batch_input[i:i + batch_size]
        rets = [generate_response(item['question'], MODEL_NAME) for item in batch]
        results = await asyncio.gather(*rets)
        # annotated_batch = [{"question": item["question"],
        #                     "trace": extract_think_content(result),
        #                     "final_answer": parse_answer(result),
        #                     "partial_solutions-annotation": result,} for item, result in zip(batch, results)]
        # annotateds.extend(annotated_batch)
        for item, result in zip(batch, results):
            messages.append({"messages": [{"role": "user", "content": item["question"]},
                                          {"role": "assistant", "content": result}]})

        # break
        # train_len1, train_len2 = int(0.9*len(annotateds)), int(0.9*len(messages))
        # with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-meta-from_ft.jsonl", "w") as file:
        #     for item in annotateds[:train_len1]:
        #         file.write(json.dumps(item) + "\n")
        with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-from_ft.jsonl", "w") as file:
            for item in messages:
                file.write(json.dumps(item) + "\n")
                
        # with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-meta-test-from_ft.jsonl", "w") as file:
        #     for item in annotateds[train_len1:]:
        #         file.write(json.dumps(item) + "\n")
        # with open(f"/disk/u/harshraj/CotIF/data/{DATA_PATH.split('/')[-1]}-llama3-hehe-partial_soln-test-from_ft.jsonl", "w") as file:
        #     for item in messages[train_len2:]:
        #         file.write(json.dumps(item) + "\n")
                    
if __name__ == "__main__":
    asyncio.run(main())