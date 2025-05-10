import re
import random
import sys
import argparse
from loguru import logger
from copy import deepcopy

import datasets
from datasets import load_dataset, concatenate_datasets
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
    upper_case_backward,
    title_case_backward,
    palindromes_backward,
    hypernyms_backward,
    paragraph_count_backward
    # translate_forward,
    ]

# backward_functions = [
#     upper_case_backward,
#     title_case_backward,
#     palindromes_backward,
#     hypernyms_backward,
#     paragraph_count_backward
#     ]

train_functions = [
    "upper_case_forward",
    "title_case_forward",
    "replace_and_ampersand_forward",
    "avoid_the_forward",
    "one_word_per_line_forward",
    "use_multiple_spaces_forward",
    "sentences_per_line_forward",
    "commas_to_semicolons_forward",
    "add_line_numbers_forward",
    "json_of_paragraphs_forward",
    "indent_paragraphs_forward",
    "insert_sentence_divider_forward",
    "render_as_html_forward",
    "upper_case_backward",
    "title_case_backward",
    "palindromes_backward",
    "hypernyms_backward"
]
eval_functions = [
    "bracket_sentences_forward",
    "highlight_logical_words_forward",
    "lower_case_forward",
    "full_stops_to_exclamation_marks_forward",
    "paragraph_count_backward"
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
    if fct != identity and cot == new_cot: 
        logger.info(f"Function {fct.__name__} did not change the CoT.")
        return False
    # if length > MAX_SEQ_LENGTH_IN_CHARS: return False
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
    if "conversations" not in datapoint:
        user = datapoint['problem'][0]
        assistant = datapoint['generated_solution'][0]
    else:
        user = datapoint['conversations'][0][0]['value']
        assistant = datapoint['conversations'][0][1]['value']
    
    # Make sure the CoT is in the R1 format.
    assistant = format_assistant_response(assistant)

    # Extract from the assistant response the part between <think> and </think>.
    try:
        cot_and_response = assistant.split('<think>')[1]
        cot, response = cot_and_response.split('</think>')
    except:
        # drop
        return {
            'messages': [],
            'applied_functions': [],
            'num_chars': []
        }
        # sys.exit(0)

    cot, response = cot.strip(), response.strip()

    # Shuffle the funcion list.    
    shuffled_functions = deepcopy(forward_functions) + deepcopy(backward_functions)
    
    # Apply the functions to the CoT and construct the list of instructions.
    conversations = []
    applied_functions = []
    num_chars = []
    # Always include the identity function.
    
    # if args.use_autoif:
    #     add = [identity] * int(args.percentage_identity*(len(shuffled_functions)+1)) + [autoif.compile]
    # else:
    #     add = [identity] * int(args.percentage_identity*len(shuffled_functions))
    # if args.use_autoif:
    #     add = [autoif.compile]
    # else:
    #     add = []
    if args.use_autoif:
        shuffled_functions = shuffled_functions + [autoif.compile]
    random.shuffle(shuffled_functions)
    
    # for f in add + shuffled_functions:
    for f in [shuffled_functions[0]]:
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
        if not new_cot and not description: 
            logger.info(f"`new_cot` or `description` is None")
            continue 
        # if f == identity:
        #     new_user = user
        #     prefix = "<think>\n"
        # else:
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
            logger.info(f"`datapoint_ok` ✅")
            # If we have applied enough functions, stop.
            # if len(conversations) == NUM_FUNCTIONS_PER_DATAPOINT:
            #     break
        else:
            logger.info(f"`datapoint_ok` ❌")
    
    # # If we didn't get enough functions applied, skip this datapoint altogether.
    # if len(conversations) < NUM_FUNCTIONS_PER_DATAPOINT:
    #     conversations = []
    #     applied_functions = []
    #     num_chars = []
        
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
    parser.add_argument("--dataset-paths", type=str, nargs="+", default=["bespokelabs/Bespoke-Stratos-17k"], help="One or more HuggingFace dataset paths (space-separated).") # open-thoughts/OpenThoughts-114k, open-thoughts/OpenThoughts2-1M
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to use from dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--output-path", type=str, default="/share/u/harshraj/CotIF/data/cotroller_dataset-mix-v4.json", help="Path to save the processed dataset JSON.")

    args = parser.parse_args()
    
    if args.use_autoif:
        autoif = AutoIf()
        logger.info("Note: `use_autoif=True` requires a 72B model. This may take time.")

    all_datasets = []
    for dataset_path in args.dataset_paths:
        dataset = load_dataset(dataset_path, split=args.split)
        all_datasets.append(dataset)
    dataset = concatenate_datasets(all_datasets)
    dataset = dataset.shuffle(seed=0)
    
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        
    dataset = dataset.map(generate_r1_prompt, remove_columns=dataset.column_names, batched=True, batch_size=1)
    
    # Create train/test split
    # split = dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = split['train']
    # test_dataset = split['test']
    
    # OOD Test
    train_dataset = dataset.filter(lambda x: x['applied_functions'] in train_functions)
    test_dataset = dataset.filter(lambda x: x['applied_functions'] in eval_functions)
    
    # train_dataset = train_dataset.shuffle(seed=0)
    
    # Save datasets
    train_dataset.to_json(args.output_path)
    test_dataset.to_json(args.output_path.replace('.json', '') + '-test.json')
    logger.info(f"Saved processed dataset to {args.output_path}")