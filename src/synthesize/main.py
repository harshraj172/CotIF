import argparse
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.synthesize.utils import save_jsonl, check_constraints, \
    tokenize_with_assistant_continuation, generate_with_batching, \
    append_constraints_to_question, constraint_templates


def using_template(dataset):
    modified_instructions = []
    for datum in tqdm(dataset, desc="Processing dataset (using templates)"):
        constraints = check_constraints(datum['response'].split("</think>", 1)[0])
        modified_instruction = append_constraints_to_question(datum['instruction'], constraints, constraint_templates)
        modified_instructions.append(modified_instruction)
        save_jsonl(modified_instructions[:int(0.9 * len(modified_instructions))], f"data/data_1k-using_templates-train-1.jsonl", append=True)
        save_jsonl(modified_instructions[int(0.9 * len(modified_instructions)):], f"data/data_1k-using_templates-test-1.jsonl", append=True)
            
def using_LLM(dataset, LLM_model, LLM_tokenizer, batch_size=20):
    # print("dataset:",dataset)
    # conversations = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing dataset in batches"):
        batch = []
        subset = dataset.select(range(i, i + batch_size))
        for datum in subset:
            constraints = check_constraints(datum['response'].split("</think>", 1)[0])
#             batch.append(tokenize_with_assistant_continuation(LLM_tokenizer, [{'role': 'user', 'content': f'''Given the below question and some structural constraints which should be present in the answer, like the number of times a word should be used, the number of sentences, or punctuation, etc. Write the constraint in the form of an question.

# ==EXAMPLE==
# ### Question: Arrange all the numbers from the list [12, 18, 36, 9] to obtain 5 using arithmetic operations such as ×, ÷, +, and -.  
# ### Constraints:  
# - The solution must be presented in exactly **three paragraphs**.  
# - Once the final answer is reached, do not verify or re-check it.  
# ### Constrained Question:  
# Arrange all the numbers from the list [12, 18, 36, 9] to obtain 5 using arithmetic operations such as ×, ÷, +, and -. Your thought process (between <think> and <\think>) should strictly adhere to the **three-paragraph** format. Ensure that you **stop immediately** once the answer is reached and do **not** verify or re-check it afterward.

# NOTE: The question will be given to an AI assistant, which will perform a chain of thought before producing the final answer. Therefore, the modified question should always remind the AI to adhere to the given constraints while performing CoT.

# ===
# ### Question: {datum['instruction']}
# ### Constraints: {constraints}
# Now, provide the revised question without any commentary.
# Note: The question should subtly incorporate the constraints. DO NOT answer the question just provide the modified question'''}, {'role': 'assistant', 'content': '### Constrained Question:'}]))

            batch.append(tokenize_with_assistant_continuation(LLM_tokenizer, [{'role': 'user', 'content': f'''Given below the structural constraints which should be present in a response, like the number of times a word should be used, the number of sentences, or punctuation, etc. Write the constraint in the form of an instruction.

==EXAMPLE==
### Constraints:  
- The solution must be presented in exactly **three paragraphs**.  
- Once the final answer is reached, do not verify or re-check it.  
### Constrained Instruction:  
Your thought process (between <think> and <\think>) should strictly adhere to the **three-paragraph** format. Ensure that you **stop immediately** once the answer is reached and do **not** verify or re-check it afterward.

NOTE: The question will be given to an AI assistant, which will perform chain of thought reasoning before producing the final response. Therefore, the modified question should always remind the AI to adhere to the given constraints while performing CoT.

===
### Constraints: {constraints}'''}, {'role': 'assistant', 'content': '### Constrained Instruction:'}]))
            
        modified_instructions = generate_with_batching(LLM_model, LLM_tokenizer, batch, skip_special_tokens=True, batch_size=len(subset), return_continuations_only=True , dont_decode_non_english=True)
        
        batch_conversations = []
        for datum, modified_instruction in zip(subset, modified_instructions): 
            batch_conversations.append([
                {"role": "user", "content": f'{modified_instruction.replace("assistant:", "").replace("Assistant:", "").replace("assistant", "").replace("Assistant", "").replace("### Constraints:", "")}\n\nNow answer this question: {datum["instruction"]}'}, 
                {"role": "assistant", "content": datum["response"]}
            ])
        save_jsonl(batch_conversations, f"data/data_1k-using_LLMs-2.jsonl", append=True)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Data Synthesis for CotIF")
    parser.add_argument(
        "--method",
        type=str,
        default="template",
        choices=["template", "llm"],
    )
    args = parser.parse_args()
    
    dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B")
    dataset = dataset["train"].select(range(6000))
        
    if args.method=="llm":
        LLM_model = AutoModelForCausalLM.from_pretrained("/share/u/models/meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
        LLM_tokenizer = AutoTokenizer.from_pretrained("/share/u/models/meta-llama/Llama-3.1-8B-Instruct")
        LLM_tokenizer.pad_token = LLM_tokenizer.eos_token
        using_LLM(dataset, LLM_model, LLM_tokenizer)
    elif args.method=="template":
        using_template(dataset)
