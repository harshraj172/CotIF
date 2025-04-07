import re
import random
import sys
import argparse
from loguru import logger
from copy import deepcopy

from datasets import load_dataset
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer

from constraints import * 

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
NUM_FUNCTIONS_PER_DATAPOINT = 8  # Same as batch size
MAX_SEQ_LENGTH_IN_CHARS = 4096 * 3  # Assume 3 chars per token

forward_functions = [
    upper_case_forward,
    lower_case_forward,
    title_case_forward,
    replace_and_ampersand_forward,
    avoid_the_forward,
    one_word_per_line_forward,
    use_multiple_spaces_forward,
    highlight_logical_words_forward,
    sentences_per_line_forward,
    commas_to_semicolons_forward,
    full_stops_to_exclamation_marks_forward,
    add_line_numbers_forward,
    json_of_paragraphs_forward,
    bracket_sentences_forward,
    indent_paragraphs_forward,
    insert_sentence_divider_forward,
    render_as_html_forward,
    translate_forward,
    ]

backward_functions = [
    upper_case_backward,
    title_case_backward,
    palindromes_backward,
    hypernyms_backward,
    paragraph_count_backward
    ]

# -1 because we also include the identity function.
assert len(forward_functions) >= NUM_FUNCTIONS_PER_DATAPOINT - 1

########################
# Function adeed by Bob:
# IMPORTANT for the DeepSeek-R1 models: the chat template strips out the CoT before training, which is bad!
# So we modify the Jinja2 template to not strip out the CoT.
########################
def get_tokenizer_with_new_chat_template(tokenizer):
    to_delete = "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
    new_template = tokenizer.get_chat_template().replace(to_delete, "")
    return AutoTokenizer.from_pretrained(MODEL, chat_template=new_template)

tokenizer = get_tokenizer_with_new_chat_template(AutoTokenizer.from_pretrained(MODEL))

def datapoint_ok(cot, new_cot, length, fct):
    if fct != identity and cot == new_cot: return False
    if length > MAX_SEQ_LENGTH_IN_CHARS: return False
    return True

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

def generate_r1_prompt(datapoint):
    user = datapoint['conversations'][0][0]['value']
    assistant = datapoint['conversations'][0][1]['value']
    
    # Make sure the CoT is in the R1 format.
    assistant = format_assistant_response(assistant)

    # Extract from the assistant response the part between <think> and </think>.
    cot_and_response = assistant.split('<think>\n')[1]
    try:
        cot, response = cot_and_response.split('</think>\n\n')
    except:
        print(cot_and_response)
        print("^^^ ERROR")
        sys.exit(0)

    cot = cot.strip()

    # Shuffle the funcion list.    
    shuffled_functions = deepcopy(forward_functions) + deepcopy(backward_functions)
    random.shuffle(shuffled_functions)
    
    # Apply the functions to the CoT and construct the list of instructions.
    conversations = []
    applied_functions = []
    num_chars = []
    # Always include the identity function.
    
    if args.use_autoif:
        add = [identity, autoif.compile]
    else:
        add = [identity]
    for f in add + shuffled_functions:
        if f is translate_forward:
            target_langs = ['fr', 'de', 'es', 'it', 'ru', 'jap', 'zh', 'ar', 'nl', 'sv']
            model_and_tokenizer = {}

            for tgt_lang in target_langs:
                model_name = f"Helsinki-NLP/opus-mt-en-{tgt_lang}"
                model_and_tokenizer[tgt_lang] = (MarianMTModel.from_pretrained(model_name).eval().cuda(), 
                                                MarianTokenizer.from_pretrained(model_name))
            result = f(cot, model_and_tokenizer)
            continue
        result = f(cot)
        new_cot = result['new_text']
        description = result['description']
        if not new_cot and not description: continue 
        if f == identity:
            new_user = user
            prefix = "<think>\n"
        else:
            new_user = user + f"\n\n## NOTES\n\nInstructions that you must follow in your reasoning chain (the part of your response before your final answer):\n- {description}"
            prefix = f"<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: I must {description}.\nIt's really important that I adhere to these instructions in my reasoning chain, since otherwise the user's need would not be met. So from the next paragraph onward, I'll {description}.\n\n"
        # assistant = f"{new_cot}\n</think>\n\n{response}"
        # conversation = [
        #         # We use no system message according to DeepSeek specs <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>.
        #         {"role": "user", "content": new_user},
        #         {"role": "assistant", "content": prefix},
        #         {"role": "assistant", "content": assistant},
        #     ]
        assistant = f"{prefix}{new_cot}\n</think>\n\n{response}"
        conversation = [
                # We use no system message according to DeepSeek specs <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>.
                {"role": "user", "content": new_user},
                {"role": "assistant", "content": assistant},
            ]
        length = len(tokenizer.apply_chat_template(conversation, tokenize=False))
        if datapoint_ok(cot, new_cot, length, f):
            conversations.append(conversation)
            applied_functions.append(f.__name__)
            num_chars.append(length)
            # If we have applied enough functions, stop.
            if len(conversations) == NUM_FUNCTIONS_PER_DATAPOINT:
                break
    
    # If we didn't get enough functions applied, skip this datapoint altogether.
    if len(conversations) < NUM_FUNCTIONS_PER_DATAPOINT:
        conversations = []
        applied_functions = []
        num_chars = []
        
    return {
        'messages': conversations,
        'applied_functions': applied_functions,
        'num_chars': num_chars
    }

# Test the functions.
def test_functions():
    datapoint = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")[0]
    with open("data/fct_test.txt", 'w') as file:
        for f in forward_functions + backward_functions:
            assistant = format_assistant_response(datapoint['conversations'][1]['value'])
            cot_and_response = assistant.split('<think>\n')[1]
            cot, response = cot_and_response.split('</think>\n\n')
            f_result = f(cot)
            new_cot = f_result['new_text']
            desc = f_result['description']
            str = f"==== {f.__name__} ====\n"
            str += f"==== I will {desc} ====\n{new_cot}\n\n"
            file.write(str)

# test_functions()
# sys.exit(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for r1 prompt training.")
    parser.add_argument("--use-autoif", action="store_true", help="Enable AutoIF (loads a 72B model).")
    parser.add_argument("--dataset-path", type=str, default="bespokelabs/Bespoke-Stratos-17k", help="HuggingFace dataset path.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to use from dataset.")
    parser.add_argument("--output-path", type=str, default="/share/u/harshraj/CotIF/data/cotroller_train_dataset-mix.json", help="Path to save the processed dataset JSON.")

    args = parser.parse_args()
    
    if args.use_autoif:
        autoif = AutoIf()
        logger.info("Note: `use_autoif=True` requires a 72B model. This may take time.")

    dataset = load_dataset(args.dataset_path, split="train")
    
    if args.num_samples:
        dataset = dataset.select(range(args.num_samples))
    
    dataset = dataset.shuffle(seed=0)

    dataset = dataset.map(generate_r1_prompt, remove_columns=dataset.features, batched=True, batch_size=1)

    dataset.to_json(args.output_path, orient="records")
    logger.info(f"Saved processed dataset to {args.output_path}")