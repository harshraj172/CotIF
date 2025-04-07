from datasets import load_dataset
from transformers import AutoTokenizer
import re
import random
import html
import sys
import json
from copy import deepcopy

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
NUM_FUNCTIONS_PER_DATAPOINT = 8  # Same as batch size
MAX_SEQ_LENGTH_IN_CHARS = 4096 * 3  # Assume 3 chars per token

##########################################################################################
# BEGIN text transformations
##########################################################################################

def identity(text):
    return {
        'description': '[no modification]',
        'new_text': text
    }

def avoid_word(text, word):
    word_cap = word.capitalize()
    match = word if word == word_cap else f'{word}/{word_cap}'
    description = f'avoid the word "{match}"'
    new_text = avoid_sentence_start(text, word.capitalize())['new_text']
    new_text = re.sub(rf'\b{word}\b ?', '', new_text, flags=re.IGNORECASE)
    return {
        'description': description,
        'new_text': new_text
    }

def avoid_sentence_start(text, word):
    word_cap = word.capitalize()
    word_upper = word.upper()
    match = word if word == word_cap else f'{word}/{word_cap}/{word_upper}'
    description = f'avoid starting any sentence with the word "{match}"'
    new_text = re.sub(rf'(^|[\.\?!:][ \n]+){word}\b. ?(.)',
                      lambda m: f"{m.group(1)}{m.group(2).capitalize()}",
                      text, flags=re.IGNORECASE)
    return {
        'description': description,
        'new_text': new_text
    }

def replace_word(text, old, new):
    old_cap = old.capitalize()
    new_cap = new.capitalize()
    old_match = old if old == old_cap else f'{old}/{old_cap}'
    new_match = new if new == new_cap else f'{new}/{new_cap}'
    description = f'avoid the word "{old_match}" and use "{new_match}" instead'
    new_text = re.sub(rf'\b{old}\b', new, text)
    new_text = re.sub(rf'\b{old_cap}\b', new_cap, new_text)
    return {
        'description': description,
        'new_text': new_text
    }

def one_word_per_line(text):
    return {
        'description': 'put each word on a separate line',
        'new_text': '\n'.join(text.split())
    }

def use_multiple_spaces(text):
    n = random.randint(2, 4)
    return {
        'description': f'always use {n} spaces between words instead of a single space',
        'new_text': re.sub(r' +', ' ' * n, text)
    }

def upper_case(text):
    return {
        'description': 'use only upper-case letters',
        'new_text': text.upper()
    }

def lower_case(text):
    return {
        'description': 'use only lower-case letters',
        'new_text': text.lower()
    }

def title_case(text):
    return {
        'description': 'start every word with a capital letter',
        'new_text': text.title()
    }

def sentences_per_line(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': 'put each sentence on a separate line',
        'new_text': "\n".join(sentences)
    }

def commas_to_semicolons(text):
    return {
        'description': 'always use semicolons instead of commas',
        'new_text': text.replace(",", ";")
    }

def full_stops_to_exclamation_marks(text):
    return {
        'description': 'always end sentences with exclamation marks instead of full stops',
        'new_text': re.sub(r'\.([ \n])', r'!\1', text)
    }

def add_line_numbers(text):
    lines = text.splitlines()
    return {
        'description': 'use line numbers (that is, prepend "1: " in front of line 1, etc.)',
        'new_text': "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    }

def json_of_paragraphs(text):
    lines = text.split("\n\n")
    json_obj = {"paragraphs": lines}
    return {
        'description': 'format the reasoning chain as a JSON object with a "paragraphs" key and a list of paragraphs as the value',
        'new_text': json.dumps(json_obj)
    }
    
def bracket_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': 'enclose each sentence in square brackets',
        'new_text': " ".join("[" + s + "]" for s in sentences)
    }

def indent_paragraphs(text):
    paragraphs = text.split("\n\n")
    return {
        'description': 'indent each paragraph with a tab character',
        'new_text': "\n\n".join("\t" + p for p in paragraphs)
    }

def highlight_logical_words(text):
    return {
        'description': f'highlight logical connector words (hmm, wait, now, thus, therefore, so, consequently, alternatively, then, okay, alright, again, but, yes) by surrounding them with asterisks',
        'new_text': re.sub(rf'\b(hmm|wait|now|thus|therefore|so|consequently|alternatively|then|okay|alright|again|but|yes)\b', r'*\1*', text, flags=re.IGNORECASE)
    }

def insert_sentence_divider(text, divider=" | "):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': f'separate all sentences with "{divider}"',
        'new_text': divider.join(sentences)
    }


def render_as_html(text):
    paragraphs = text.split("\n\n")
    html_paragraphs = "".join(
        "<p>" + html.escape(p).replace('\n', '<br>') + "</p>" for p in paragraphs
    )
    html_output = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset='UTF-8'>\n"
        "  <title>Reasoning chain</title>\n"
        "</head>\n"
        "<body>\n"
        f"{html_paragraphs}\n"
        "</body>\n"
        "</html>"
    )
    return {
        'description': 'render the reasoning chain as HTML by escaping HTML special characters, wrapping paragraphs in <p> tags, and replacing newline characters with <br> tags within paragraphs',
        'new_text': html_output
    }

def replace_and_ampersand(text): return replace_word(text, 'and', '&')
def avoid_the(text): return avoid_word(text, 'the')

##########################################################################################
# END text transformations
##########################################################################################

functions = [
    upper_case,
    lower_case,
    title_case,
    replace_and_ampersand,
    avoid_the,
    one_word_per_line,
    use_multiple_spaces,
    highlight_logical_words,
    sentences_per_line,
    commas_to_semicolons,
    full_stops_to_exclamation_marks,
    add_line_numbers,
    json_of_paragraphs,
    bracket_sentences,
    indent_paragraphs,
    insert_sentence_divider,
    render_as_html,
    ]

# -1 because we also include the identity function.
assert len(functions) >= NUM_FUNCTIONS_PER_DATAPOINT - 1

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
    shuffled_functions = deepcopy(functions)
    random.shuffle(shuffled_functions)
    
    # Apply the functions to the CoT and construct the list of instructions.
    conversations = []
    applied_functions = []
    num_chars = []
    # Always include the identity function.
    for f in [identity] + shuffled_functions:
        result = f(cot)
        new_cot = result['new_text']
        description = result['description']
        if f == identity:
            new_user = user
            prefix = "<think>\n"
        else:
            new_user = user + f"\n\n## NOTES\n\nInstructions that you must follow in your reasoning chain (the part of your response before your final answer):\n- {description}"
            prefix = f"<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: I must {description}.\nIt's really important that I adhere to these instructions in my reasoning chain, since otherwise the user's need would not be met. So from the next paragraph onward, I'll {description}.\n\n"
        assistant = f"{new_cot}\n</think>\n\n{response}"
        conversation = [
                # We use no system message according to DeepSeek specs <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>.
                {"role": "user", "content": new_user},
                {"role": "assistant", "content": prefix},
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
        for f in functions:
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

# Load dataset from the hub
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
dataset = dataset.shuffle(seed=42)

# Convert our dataset to the r1 prompt
dataset = dataset.map(generate_r1_prompt, remove_columns=dataset.features, batched=True, batch_size=1)

# Save datasets to disk 
dataset.to_json("data/forward/cotroller_train_dataset.json", orient="records")
