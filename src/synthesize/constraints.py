import re
import random
import html
import numpy as np
import signal
from tqdm import tqdm
import json
import jsonlines
from collections import Counter
from concurrent.futures import TimeoutError

from nltk.corpus import wordnet as wn
import spacy 
from string import punctuation

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM,\
    AutoTokenizer

spacy_nlp = spacy.load('en_core_web_sm')
random.seed(0)

numbering_list = ['3', '7)', '7.', '4', 'iii.', 'iii-', '8.', '4-', 'v:', 'I:', 'ii.', 'i.', 'V)', 'E)', 'I)', 'III.', 'III)', '2-', '1)', 'v-', 'III', 'I.', 'c)', '1.', 'V-', 'iv)', 'A)', 'v)', 'IV', 'C.', 'ii)', 'I', 'IV.', 'C)', 'II-', '2.', 'III-', 'IV)', 'd)', 'iii', 'i-', 'iii:', 'A.', 'B.', '1', '6)', 'ii', '8)', '3)', 'e)', 'ii-', '5-', 'II)', 'iv-', '2)', 'e.', 'IV:', 'III:', 'i)', '10.', 'V', 'V.', 'v.', 'D)', 'E.', 'iv:', 'B)', 'II', 'ii:', 'V:', 'a.', '5.', 'IV-', '9.', 'D.', '3.', '4:', '2:', 'i', 'II.', '3-', '2', 'c.', 'a)', '3:', '10)', 'd.', 'i:', 'iv.', '1-', '4.', '5', 'iv', 'iii)', 'b.', '1:', 'II:', 'v', '5:', '6.', 'b)', 'I-', '9)', '4)', '5)']

stopwords_list = ['es', 'ing', 'ed', 'include', 'includes', 'also', 'haven', 'are', 'why', 'most', "won't", 'against', 'with', 'needn', 'couldn', 'now', 'mustn', 'who', 'under', 'doing', 'am', 'aren', 'they', "didn't", 'd', 'doesn', 'if', 'he', 'her', "haven't", 'isn', 'own', 'does', 'such', 'until', 'into', 'had', 'again', 'over', "hadn't", "you'll", 't', 'by', 'be', "wasn't", 'so', 'yours', 'both', 'any', 'did', "you've", 'these', 'myself', 'o', 'hasn', "isn't", 'you', 'other', 'shan', 'being', 'yourselves', 'was', 'no', 'm', 'those', 'will', 'its', 'itself', 'have', 'down', 'weren', 'having', 'wouldn', 'herself', "mustn't", 'very', 'do', "should've", 'him', "you'd", 'below', 'just', 'that', 'for', 'which', 'but', 'nor', 'all', 'then', 'i', 'whom', 'it', 'once', 'here', 've', "you're", 'ours', "that'll", 'a', 'won', 'himself', 'where', 'this', 'your', "hasn't", 'same', 'when', 'ourselves', 'because', "needn't", 'theirs', 'from', 'mightn', 'my', 'while', 'yourself', "she's", 'each', "doesn't", 'only', 'at', 's', 'their', "wouldn't", 'shouldn', 'and', 'themselves', 'hers', 'has', 'up', 'ma', 'in', 'll', 'we', 're', 'y', 'of', 'after', 'our', "shan't", 'before', 'wasn', 'can', 'should', 'been', 'through', 'as', 'further', 'during', 'between', 'there', 'me', 'on', 'don', "shouldn't", 'more', 'out', "don't", 'the', "weren't", "aren't", "it's", 'what', 'or', "couldn't", 'hadn', "mightn't", 'his', 'above', 'to', 'how', 'few', 'off', 'them', 'didn', 'ain', 'not', 'she', 'an', 'than', 'too', 'is', 'some', 'were', 'about']

common_title_words_set = {'introduction', 'conclusion', 'section', 'chapter', 'works', 'notes', 'note', 'further', 'see', 'references', 'reference', 'section', 'title', 'conclusion', 'intro', 'introduction', 'executive', 'summary', 'key', 'plot', 'theme'}
stopwords_set = set(stopwords_list + numbering_list)

def identity(text):
    return {
        'description': '[no modification]',
        'new_text': text
    }

# avoid words
## forward
def avoid_sentence_start(text, word):
    word_cap = word.capitalize()
    word_upper = word.upper()
    match = word if word == word_cap else f'{word}/{word_cap}/{word_upper}'
    description = random.choice(
        [f'Avoid starting any sentence with the word "{match}".',
        f'Make sure no sentence begins with "{match}".',
        f'Do not start a sentence using the word "{match}".',
        f'The word "{match}" should not be at the beginning of any sentence.',
        f'Ensure that no sentence starts with "{match}".',
        f'"{match}" must not be used as the first word of any sentence.']
    )
    new_text = re.sub(rf'(^|[\.\?!:][ \n]+){word}\b. ?(.)',
                      lambda m: f"{m.group(1)}{m.group(2).capitalize()}",
                      text, flags=re.IGNORECASE)
    return {
        'description': description,
        'new_text': new_text
    }
def avoid_word(text, word):
    word_cap = word.capitalize()
    match = word if word == word_cap else f'{word}/{word_cap}'
    description = random.choice(
        [f'Avoid the word "{match}".',
        f'Do not use the word "{match}" in your response.',
        f'Try to exclude the word "{match}".',
        f'Make sure the word "{match}" is not included.',
        f'Your output should not contain the word "{match}".',
        f'Refrain from using "{match}" anywhere in the text.']
    )
    new_text = avoid_sentence_start(text, word.capitalize())['new_text']
    new_text = re.sub(rf'\b{word}\b ?', '', new_text, flags=re.IGNORECASE)
    return {
        'description': description,
        'new_text': new_text
    }
def avoid_the_forward(text): return avoid_word(text, 'the')


## backward
def include_word_forward(text):
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count word frequencies
    word_counts = Counter(words)
    most_common_word, freq = word_counts.most_common(1)[0]

    description = random.choice([f'include the word "{most_common_word}" (most frequent word in the text, appears {freq} times)',
                                f'Make sure to include the most frequent word "{most_common_word}", which appears {freq} times.',
                                f'The word "{most_common_word}" occurs {freq} times and is the most common—include it in the output.',
                                f'Include "{most_common_word}" in the result; it’s the most frequently used word with {freq} occurrences.',
                                f'"{most_common_word}" shows up {freq} times in the text, more than any other word. Use it.',
                                f'Your response should contain the word "{most_common_word}", which has the highest frequency: {freq} times.'])
    
    return {
        'description': description,
        'new_text': text
    }


# replace word
## forward
def replace_word(text, old, new):
    old_cap = old.capitalize()
    new_cap = new.capitalize()
    old_match = old if old == old_cap else f'{old}/{old_cap}'
    new_match = new if new == new_cap else f'{new}/{new_cap}'
    description = random.choice(
        [f'Avoid using the word "{old_match}" and replace it with "{new_match}".',
        f'Use "{new_match}" instead of "{old_match}".',
        f'Do not use "{old_match}"; go with "{new_match}" instead.',
        f'Substitute "{old_match}" with "{new_match}".',
        f'"{old_match}" should be avoided—use "{new_match}" in its place.',
        f'Replace "{old_match}" with the preferred word "{new_match}".']
    )
    new_text = re.sub(rf'\b{old}\b', new, text)
    new_text = re.sub(rf'\b{old_cap}\b', new_cap, new_text)
    return {
        'description': description,
        'new_text': new_text
    }
def replace_and_ampersand_forward(text): return replace_word(text, 'and', '&')


# one word per line
## forward
def one_word_per_line_forward(text):
    return {
        'description': random.choice(
                        ['Put each word on a separate line.',
                        'Write every word on its own line.',
                        'Each word should appear on a new line.',
                        'Make sure every word is printed on a separate line.',
                        'Separate the words by placing each one on its own line.',
                        'Display each word line by line.']
                    ),
        'new_text': '\n'.join(text.split())
    }


# use multiple spaces
## forward
def use_multiple_spaces_forward(text):
    n = random.randint(2, 4)
    return {
        'description': random.choice(
                        [f'Always use {n} spaces between words instead of just one.',
                        f'Replace single spaces with {n} spaces between each word.',
                        f'Ensure there are exactly {n} spaces between all words.',
                        f'Use {n} spaces to separate words, not just one.',
                        f'Make sure to insert {n} spaces between words rather than a single space.',
                        f'Words should be separated by {n} spaces, not the usual one.']
                    ),
        'new_text': re.sub(r' +', ' ' * n, text)
    }


# upper case
## forward
def upper_case_forward(text):
    return {
        'description': random.choice(
                    ['Use only upper-case letters.',
                    'Convert all letters to upper-case.',
                    'Make sure everything is written in capital letters.',
                    'Text should be entirely in upper-case.',
                    'All characters must be upper-case.',
                    'Write the output using only capital letters.']
                ),
        'new_text': text.upper()
    }
    
## backward
def upper_case_backward(text):
    upper_case_words = [word for word in text.split() if word.isupper()]
    if upper_case_words:
        description = random.choice(
                    [f'Always use these words in uppercase: {"".join(upper_case_words[:3])}',
                    f'Make sure the following words appear in all caps: {"".join(upper_case_words[:3])}',
                    f'The words {"".join(upper_case_words[:3])} should always be written in uppercase.',
                    f'Use uppercase letters for these specific words: {"".join(upper_case_words[:3])}',
                    f'Ensure that {"".join(upper_case_words[:3])} are written entirely in capital letters.',
                    f'These words must be capitalized fully: {"".join(upper_case_words[:3])}']
                )
        new_text = ''
        for word in text.split():
            if word.strip(punctuation).upper() in upper_case_words:
                new_text += ' ' + word.upper()
            else:
                new_text += ' ' + word
        new_text = new_text.strip()
    else:
        description, new_text = None, None
    return {'description': description, 'new_text':new_text}


# lower case
## forward
def lower_case_forward(text):
    return {
        'description': random.choice(
                        ['Use only lower-case letters.',
                        'Convert all letters to lower-case.',
                        'Make sure everything is written in small letters.',
                        'Text should be entirely in lower-case.',
                        'All characters must be lower-case.',
                        'Write the output using only lower-case letters.']
                    ),
        'new_text': text.lower()
    }


# title case
## forward
def title_case_forward(text):
    return {
        'description': random.choice(
                    ['Start every word with a capital letter.',
                    'Capitalize the first letter of each word.',
                    'Each word should begin with an uppercase letter.',
                    'Make sure the first character of every word is capitalized.',
                    'Write every word in title case.',
                    'Use capital letters at the start of all words.']
                ),
        'new_text': text.title()
    }

## backward
def title_case_backward(text):
    upper_case_words = [word for word in text.split() if word.istitle()]
    if upper_case_words:
        description = random.choice(
                    [f'Always use these words in title case: {"".join(upper_case_words[:3])}',
                    f'Make sure the following words appear in title case: {"".join(upper_case_words[:3])}',
                    f'The words {"".join(upper_case_words[:3])} should each start with a capital letter.',
                    f'Use title case for these words: {"".join(upper_case_words[:3])}',
                    f'Ensure that each of the following words is in title case: {"".join(upper_case_words[:3])}',
                    f'These words must appear in title case format: {"".join(upper_case_words[:3])}']
                )
        new_text = ''
        for word in text.split():
            if word.strip(punctuation).title() in upper_case_words:
                new_text += ' ' + word.title()
            else:
                new_text += ' ' + word
        new_text = new_text.strip()
    else:
        description, new_text = None, None
    return {'description': description, 'new_text':new_text}


# sentence per line
## forward
def sentences_per_line_forward(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': random.choice(
                ['Put each sentence on a separate line.',
                'Write every sentence on its own line.',
                'Each sentence should appear on a new line.',
                'Make sure each sentence is placed on a separate line.',
                'Separate the sentences by writing each one on a different line.',
                'Display each sentence line by line.']
            ),
        'new_text': "\n".join(sentences)
    }


# commas to semicolon
## forward
def commas_to_semicolons_forward(text):
    return {
        'description': random.choice(
                    ['Always use semicolons instead of commas.',
                    'Replace all commas with semicolons.',
                    'Use semicolons in place of commas.',
                    'Commas should be swapped out for semicolons.',
                    'Make sure to use semicolons where commas would normally go.',
                    'Do not use commas; use semicolons instead.']
                ),
        'new_text': text.replace(",", ";")
    }


# change full stops to exclamation marks
## forward
def full_stops_to_exclamation_marks_forward(text):
    return {
        'description': random.choice(
                    ['Always end sentences with exclamation marks instead of full stops.',
                    'Replace all full stops with exclamation marks at the end of sentences.',
                    'Use exclamation marks to end every sentence, not full stops.',
                    'Every sentence should end with an exclamation mark rather than a period.',
                    'Do not use full stops—use exclamation marks to finish sentences.',
                    'Make sure to end all sentences with exclamation marks instead of periods.']
                ),
        'new_text': re.sub(r'\.([ \n])', r'!\1', text)
    }


# add line numbers
## forward
def add_line_numbers_forward(text):
    lines = text.splitlines()
    return {
        'description': random.choice(
                    ['Use line numbers (i.e., add "1: " before line 1, and so on).',
                    'Prepend each line with its line number, like "1: " for the first line.',
                    'Add line numbers at the start of each line (e.g., "1: ").',
                    'Each line should begin with its corresponding number, such as "1: " for line 1.',
                    'Prefix every line with its line number—for example, "1: ", "2: ", etc.',
                    'Number each line by placing "1: ", "2: ", etc.], at the beginning.']
                ),
        'new_text': "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    }
    
def contains_list_or_enumeration_forward(text):
    if (re.match(r"\b(first|second|third|one|two|three|1\.|2\.|3\.)\b", text, re.IGNORECASE)):
        description = random.choice(
                    ['Use line numbers or enumeration in your reasoning process.',
                    'Structure your reasoning with numbered steps or lines.',
                    'Include line numbers or use a numbered list when explaining.',
                    'Present your reasoning in an enumerated or line-numbered format.',
                    'Break down your reasoning into numbered parts or steps.',
                    'Organize your explanation using enumeration or line numbers.']
                )
        new_text = text
    else:
        description, new_text = None, None
    return {'description': description, 'new_text': new_text}



# format the paragraphs as json
## forward
def json_of_paragraphs_forward(text):
    lines = text.split("\n\n")
    json_obj = {"paragraphs": lines}
    return {
        'description': random.choice(
                    ['Format the reasoning chain as a JSON object with a "paragraphs" key and a list of paragraphs as its value.',
                    'Represent your reasoning as a JSON object where "paragraphs" maps to a list of paragraph strings.',
                    'Output the reasoning in JSON format, using "paragraphs" as the key for a list of paragraphs.',
                    'Structure the explanation as a JSON object: the key should be "paragraphs", and the value should be a list of paragraphs.',
                    'Provide the reasoning as a JSON object with a single key "paragraphs" pointing to a list of paragraph texts.',
                    'Wrap the reasoning chain in a JSON object using "paragraphs" as the key and the paragraphs themselves as a list.']
                ),
        'new_text': json.dumps(json_obj)
    }


# bracket the sentences
## forward
def bracket_sentences_forward(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': random.choice(
                    ['Enclose each sentence in square brackets.',
                    'Wrap every sentence with [ and ].',
                    'Each sentence should be surrounded by square brackets.',
                    'Put square brackets around every sentence.',
                    'Make sure every sentence is enclosed in [brackets].',
                    'Use square brackets to frame each sentence.']
                ),
        'new_text': " ".join("[" + s + "]" for s in sentences)
    }


# indent the paragraphs with a tab 
## forward
def indent_paragraphs_forward(text):
    paragraphs = text.split("\n\n")
    return {
        'description': random.choice(
                    ['Indent each paragraph with a tab character.',
                    'Start every paragraph with a tab for indentation.',
                    'Each paragraph should begin with a tab character.',
                    'Use a tab to indent all paragraphs.',
                    'Make sure every paragraph is indented using a tab.',
                    'Prefix each paragraph with a tab character for indentation.']
                ),
        'new_text': "\n\n".join("\t" + p for p in paragraphs)
    }


# highlight the logical words
## forward
def highlight_logical_words_forward(text):
    return {
        'description': random.choice(
                    [f'Highlight logical connector words (hmm, wait, now, thus, therefore, so, consequently, alternatively, then, okay, alright, again, but, yes) by surrounding them with asterisks.',
                    f'Surround logical connector words like (hmm, wait, now, thus, etc.) with asterisks to highlight them.',
                    f'Use asterisks to mark logical connectors such as hmm, thus, therefore, so, and similar words.',
                    f'Words like hmm, wait, now, thus, and so on should be wrapped in asterisks to emphasize their logical role.',
                    f'Add asterisks around logical linking words (e.g., hmm, thus, consequently, etc.) to highlight them.',
                    f'Emphasize logical connectors—like "but", "therefore", and "then"—by enclosing them in asterisks.']
                ),
        'new_text': re.sub(rf'\b(hmm|wait|now|thus|therefore|so|consequently|alternatively|then|okay|alright|again|but|yes)\b', r'*\1*', text, flags=re.IGNORECASE)
    }


# separate sentences with a divider
## forward
def insert_sentence_divider_forward(text, divider=" | "):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': random.choice(
                    [f'Separate all sentences with "{divider}".',
                    f'Use "{divider}" to divide each sentence.',
                    f'Insert "{divider}" between every sentence.',
                    f'Each sentence should be separated using "{divider}".',
                    f'Make sure to place "{divider}" between sentences.',
                    f'Join the sentences using "{divider}" as the separator.']
                ),
        'new_text': divider.join(sentences)
    }


# render the text as html
## forward
def render_as_html_forward(text):
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
        'description': random.choice(
                    ['Render the reasoning chain as HTML: escape special characters, wrap each paragraph in <p> tags, and replace newlines within paragraphs with <br> tags.',
                    'Convert the reasoning into HTML by escaping special characters, using <p> tags for paragraphs, and inserting <br> for line breaks inside paragraphs.',
                    'Output the reasoning as HTML—escape HTML characters, enclose paragraphs in <p> tags, and turn newlines into <br> tags.',
                    'Format the reasoning chain in HTML: paragraphs should be wrapped with <p> tags, newlines within them turned into <br>, and special HTML characters escaped.',
                    'Escape HTML special characters, use <p> tags to wrap each paragraph, and replace all newlines within paragraphs with <br> tags.',
                    'Generate the reasoning as HTML with proper escaping, <p> tags for paragraphs, and <br> tags in place of newlines within them.']
                ),
        'new_text': html_output
    }


# palindromes
## backward
def palindromes_backward(text):
    words = text.split()
    palindromes = [
        word.strip(punctuation)
        for word in words
        if len(word.strip(punctuation)) > 3 and word.strip(punctuation).lower() == word.strip(punctuation).lower()[::-1]
    ]
    if palindromes:
        description = random.choice(
                    [f'In your reasoning chain, use the palindrome words: {"".join(palindromes)}.',
                    f'Make sure your reasoning includes the following palindromes: {"".join(palindromes)}.',
                    f'Include these palindrome words in your explanation: {"".join(palindromes)}.',
                    f'Your reasoning should contain the palindrome words: {"".join(palindromes)}.',
                    f'Use the following palindromic words as part of your reasoning: {"".join(palindromes)}.',
                    f'Ensure the words {"".join(palindromes)} — all palindromes — appear in your reasoning chain.']
                )
        new_text = text
    else:
        new_text, description = None, None
        
    return {'description': description, 'new_text': new_text}


# common hypernyms
## backward
def strip_left_stopwords(e_text):
  """
  Removes common stopwords from the left side of a text until a significant word is found.

  Args:
      e_text (str): The text to strip from the left side.

  Returns:
      str: Text with left-side stopwords removed.
  """
  e_text2 = []
  add_rest = False
  for et in e_text.split():
      if add_rest or ((et.lower() not in stopwords_set and et.lower() not in common_title_words_set) or et.lower().strip(".") in {"a", "an", "united", "the", "new", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",  "asian", "american", "african", "european", }):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)

def analyze_nouns_common_hypernyms(text):
    doc = spacy_nlp(text)
    # candidate_nouns = [strip_left_stopwords(e.text)  for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set]        
    candidate_nouns = [
        token.lemma_ for token in doc 
        if token.pos_ == "NOUN" and token.lemma_.lower() not in stopwords_set and len(token.text) > 2
    ]
    noun_synsets = {}
    for noun in candidate_nouns:
        syns = wn.synsets(noun, "n")
        if syns:
            noun_synsets[noun] = syns[0]

    hypernym_map = {}
    noun_list = sorted(noun_synsets.keys())
    for i in range(len(noun_list)):
        for j in range(i+1, len(noun_list)):
            syn_i = noun_synsets[noun_list[i]]
            syn_j = noun_synsets[noun_list[j]]
            lch = syn_i.lowest_common_hypernyms(syn_j)
            if lch:
                for common_hyp in lch:
                    name = common_hyp.lemmas()[0]
                    if name.name() in {"act", "whole", "communication", "thing", "content", "group", "social group", "unit", "organism", "region", "area", "geographic point", "relation", "attribute", "happening", "action", "entity", "causal agent", 'administrative district', "abstraction", "object", "point", "event", "physical entity", "matter", "part", "person", "state"}:
                        continue
                    hypernym_map.setdefault(name, set()).update([noun_list[i], noun_list[j]])
    
    hypernym_map = {
        hyp: list(nset) for hyp, nset in hypernym_map.items() if len(nset) >= 2
    }

    return hypernym_map

def hypernyms_backward(text):
    hypernym_map = analyze_nouns_common_hypernyms(text)
    if hypernym_map:
        noun_type, nouns = random.sample(list(hypernym_map.items()), 1)[0]
        description = random.choice(
                    [f'In your chain of reasoning, include objects related to {noun_type}, such as {" ".join(nouns)}.',
                    f'Your reasoning should involve {noun_type}-related objects like {" ".join(nouns)}.',
                    f'Make sure to mention things connected to {noun_type} — for example: {" ".join(nouns)}.',
                    f'Use objects associated with {noun_type} in your reasoning, like {" ".join(nouns)}.',
                    f'In your explanation, refer to items related to {noun_type}, such as {" ".join(nouns)}.',
                    f'Ensure that your reasoning chain includes examples of {noun_type} objects like {" ".join(nouns)}.']
                )
        new_text = text
    else:
        description, new_text = None, None
    return {'description': description, 'new_text': new_text}


# paragraph count
## backward
def paragraph_count_backward(text):
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paras) > 0:
        description = random.choice(
                    [f'Use {len(paras)} paragraphs during your reasoning process.',
                    f'Your reasoning should consist of {len(paras)} distinct paragraphs.',
                    f'Ensure your explanation is broken into {len(paras)} parts.',
                    f'Include exactly {len(paras)} paragraphs in your reasoning chain.',
                    f'Organize your reasoning using {len(paras)} separate paragraphs.',
                    f'Break down your reasoning into {len(paras)} clearly defined sections.']
                )
        new_text = text
    else:
        description, new_text = None, None
    return {'description': description, 'new_text': new_text}


# translation
## forward
# target_langs = ['fr', 'de', 'es', 'it', 'ru', 'jap', 'zh', 'ar', 'nl', 'sv']
LANG_CODE_TO_NAME = {'fr': 'French', 'de': 'German', 'es': 'Spanish', 'it': 'Italian', 'ru': 'Russian', 'jap': 'Japanese', 'zh': 'Chinese', 'ar': 'Arabic', 'nl': 'Dutch', 'sv': 'Swedish'}
# model_and_tokenizer = {}

# for tgt_lang in target_langs:
#     model_name = f"Helsinki-NLP/opus-mt-en-{tgt_lang}"
#     model_and_tokenizer[tgt_lang] = (MarianMTModel.from_pretrained(model_name), 
#                                     MarianTokenizer.from_pretrained(model_name))

def translate_forward(text, model_and_tokenizer):
    target_langs = ['fr', 'de', 'es', 'it', 'ru', 'jap', 'zh', 'ar', 'nl', 'sv']

    tgt_lang = random.choice(target_langs)
    model, tokenizer = model_and_tokenizer[tgt_lang]

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return {'description': random.choice(
                        [f'For your reasoning chain, use "{LANG_CODE_TO_NAME[tgt_lang]}" only.',
                        f'Your entire explanation should be in "{LANG_CODE_TO_NAME[tgt_lang]}" language.',
                        f'Produce the reasoning exclusively in "{LANG_CODE_TO_NAME[tgt_lang]}".',
                        f'Make sure to write your reasoning solely in "{LANG_CODE_TO_NAME[tgt_lang]}" language.',
                        f'The reasoning must be written using "{LANG_CODE_TO_NAME[tgt_lang]}" and no other language.',
                        f'Use only "{LANG_CODE_TO_NAME[tgt_lang]}" in your chain of reasoning—no code-switching.']
                    ),
            'new_text': translated_text}


# replace words with emojis
## forward
def replace_words_with_emojis(text):
    emoji_map = {
        'error': '❌',
        'warning': '⚠️',
        'analysis': '🔍',
        'result': '✅',
        'success': '🎉',
        'failure': '💔',
        'logic': '🤔',
        'calculate': '🧮',
        'problem': '❓',
        'solution': '💡',
        'reasoning': '🧠',
        'trace': '📝',
        'thought': '💭',
        'data': '📊',
        'compute': '💻',
        'model': '🤖',
        'optimization': '⚙️',
        'test': '🧪',
        'experiment': '🔬',
        'input': '📥',
        'output': '📤',
        'unknown': '❓',
        'uncertain': '🤷',
        'prompt': '💬',
        'instruct': '📝'
    }
    
    pattern = r'\b(' + '|'.join(re.escape(word) for word in emoji_map.keys()) + r')\b'
    
    replaced = {}
    
    def replacer(match):
        original = match.group(0)
        lower_word = original.lower()
        emoji = emoji_map[lower_word]
        replaced.setdefault(original, emoji)
        return emoji
    
    new_text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
    
    description = random.choice(
                    ["In your reasoning chain, replace the words {} with their corresponding emojis {} respectively.".format(list(replaced.keys()), list(replaced.values())),
                    "Swap out the words {} for these emojis {} in your explanation.".format(list(replaced.keys()), list(replaced.values())),
                    "Replace each of the following words {} with its emoji equivalent {}.".format(list(replaced.keys()), list(replaced.values())),
                    "Use emojis {} in place of the words {} throughout your reasoning.".format(list(replaced.values()), list(replaced.keys())),
                    "Substitute the words {} with the matching emojis {} during your explanation.".format(list(replaced.keys()), list(replaced.values())),
                    "Convert the words {} into these emojis {} in your reasoning chain.".format(list(replaced.keys()), list(replaced.values()))]
                )
    
    return {'description': description, 'new_text': new_text}


# Qwen's autoif
def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class AutoIf:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-72B-Instruct", device_map="auto").half().eval()
        
        self.tokenizer_nli = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
        self.model_nli = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7").eval().cuda()
        
        self.seed_instructions = [each.strip() for each in open("/share/u/harshraj/CotIF/data/seed_instruction.txt").readlines()]
        self.generated_eval_functions = []
        self.filtered_generated_eval_functions = []
        self.generated_instructions = []
        self.filtered_generated_instructions = []
        self.generated_responses = []
        self.filtered_generated_responses = []
        self.filtered2_generated_responses = []
        
    def compile(self, text):
        self.generate_seed()
        self.generate_eval_function()
        self.filter_generated_eval_function()
        self.generate_instruction()
        self.filter_generated_instruction()
        self.generate_response(text)
        self.filter_generated_response()
        self.filter2_generated_response()
        return self.filtered2_generated_responses
        
    def generate_seed(self, k=2):
        if k <= 0:
            return self.seed_instructions
        
        augment_instruction_prompt = """You are an expert for writing instructions. Please provide 50 different instructions that meet the following requirements:
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

        augment_instructions = augment_instruction_prompt.format(seed_instructions='\n'.join(self.seed_instructions))
        
        input_ids = self.tokenizer.encode(augment_instructions, return_tensors="pt").cuda()
        outputs = self.model.generate(input_ids, max_length=1024, do_sample=True, temperature=0.7)
        generated_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        new_seeds = [line.strip() for line in generated_text.split('\n') if line.strip()]
        self.seed_instructions = self.seed_instructions + new_seeds
        
        random.shuffle(self.seed_instructions)
        return self.generate_seed(k - 1)
    
    def generate_eval_function(self, k=2):
        prompt_template = (
            "You are an expert for writing evaluation functions in Python to evaluate whether a response strictly follows an instruction.\n"
            "Here is the instruction: {instruction}\n"
            "Please write a Python function named `evaluate` to evaluate whether an input string `response` follows this instruction. "
            "If it follows, simply return True, otherwise return False.\n"
            "Please response with a single JSON includes the evaluation function in the key `func`, and a list of three test cases in the key `cases`, "
            "which includes an input in the key `input` and an expected output in the key `output` in (true, false).\n"
            "Here is an example of output JSON format: {{\"func\": JSON_STR(use only \\n instead of \n), \"cases\": [{{\"input\": str, \"output\": str}}]}}."
        )

        for instruction in self.seed_instructions:
            prompt = prompt_template.format(instruction=instruction)
            for _ in range(k):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
                outputs = self.model.generate(input_ids, max_length=1024, do_sample=True, temperature=0.7)
                generated_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                self.generated_eval_functions.append({
                    "prompt": prompt,
                    "instruction": instruction,
                    "gpt-answer": generated_text
                })
                
    def filter_generated_eval_function(self):
        collect_packages = []
        for result in self.generated_eval_functions:
            res = result['gpt-answer']
            eval_funcs, test_cases = [], []
            for each in res:
                try:
                    json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
                except IndexError:
                    continue
                try:
                    res_dict = json.loads(json_dict)
                except json.JSONDecodeError:
                    continue
                func = res_dict['func']
                if '\\n' in func:
                    func = func.replace('\\n', '\n')
                try:
                    exec(func)
                except Exception:
                    continue
                for line in func.split('\n'):
                    if 'import' in line or 'download' in line or 'requests' in line:
                        collect_packages.append(line)
        print(list(set(collect_packages)))

        for result in tqdm(self.generated_eval_functions):
            res = result['gpt-answer']
            eval_funcs, test_cases = [], []
            for each in tqdm(res):
                try:
                    json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
                except IndexError:
                    continue
                try:
                    res_dict = json.loads(json_dict)
                except json.JSONDecodeError:
                    continue

                # func rejection and cleaning
                func = res_dict['func'].strip()
                func = '\n'.join([line for line in func.split('\n') if 'download' not in line and 'requests' not in line])
                try:
                    exec(func)
                except Exception:
                    continue
                eval_funcs.append(func)

                for each_case in res_dict['cases']:
                    try:
                        test_cases.append((each_case['input'], each_case['output']))
                    except KeyError:
                        print(each_case)

            eval_funcs = list(set(eval_funcs))
            test_cases = list(map(json.loads, set(map(json.dumps, test_cases))))
            if len(eval_funcs) < 3 or len(test_cases) < 10:
                continue

            filtered_test_cases = []
            for each in tqdm(test_cases):
                flag = False
                for func in eval_funcs:
                    local_vars = {}
                    try:
                        exec(func, globals(), local_vars)
                    except Exception:
                        continue
                    if 'evaluate' not in local_vars:
                        continue
                    eval_func = local_vars['evaluate']
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res_val = eval_func(each[0])
                    except Exception:
                        res_val = None
                    finally:
                        signal.alarm(0)
                    if res_val is not None and res_val == each[1]:
                        flag = True
                if flag:
                    filtered_test_cases.append(each)

            scored_funcs = []
            for func in tqdm(eval_funcs):
                local_vars = {}
                try:
                    exec(func, globals(), local_vars)
                except Exception:
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
                    except Exception:
                        res_val = None
                    finally:
                        signal.alarm(0)
                    if res_val is None or res_val != out:
                        acc.append(0)
                    else:
                        acc.append(1)
                acc = np.mean(acc) if acc else 0
                scored_funcs.append((func, acc))
            valid_funcs = [each for each in scored_funcs if each[1] >= 0.8]
            if not valid_funcs:
                continue

            self.filtered_generated_eval_functions.append({
                "instruction": result['instruction'],
                "eval_func": valid_funcs,
                "cases": filtered_test_cases
            })
            

    def generate_instruction(self, k=2):
        count = 0
        filter_count = 0

        for line in tqdm(self.filtered_generated_eval_functions, desc="Generating back-translated instructions"):
            funcs = line["eval_func"][:3]

            instruction_prompt = f"""You are an expert in converting the Python eval function code into the corresponding instruction text. I will provide the eval function code. Please strictly follow the code to convert it into the corresponding instruction text. Here's an example: 

[["def evaluate(response):\n    return 'e' not in response.lower()", 1.0], ["def evaluate(response):\n    words = response.split()\n    for word in response.split():\n        if 'e' in word.lower():\n            return False\n    return True", 1.0], ["def evaluate(response):\n    return all('e' not in word.lower() for word in response.split())", 1.0]] 

["Answer without using any words that contain the letter 'E'.","Answer with words that do not contain the letter 'E'.","Answer with words that do not contain the letter 'E'."] Please convert the following eval function into instructions stored in a list: 

{funcs}"""

            for _ in range(k):
                input_ids = self.tokenizer.encode(instruction_prompt, return_tensors="pt").cuda()
                outputs = self.model.generate(input_ids, max_length=1024, do_sample=True, temperature=0.7)
                generated_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

                try:
                    back_instruction = json.loads(generated_text)
                    break
                except Exception:
                    filter_count += 1
                    continue

            line["back_instruction"] = back_instruction
            self.generated_instructions.append(line)
            count += 1
            

    def filter_generated_instruction(self):
        count = 0 
        filter_count = 0

        for line in tqdm(self.generated_instructions, desc="Filtering back-translated instructions"):
            back_instructions = line["back_instruction"]
            ori_ins = line["instruction"]

            nli_scores = []
            for back_ins in back_instructions[:3]:
                premise = ori_ins
                hypothesis = back_ins

                inputs = self.tokenizer_nli(premise, hypothesis, truncation=True, return_tensors="pt")
                output = self.model_nli(inputs["input_ids"].cuda())
                prediction = torch.softmax(output["logits"][0], -1).tolist()
                label_names = ["entailment", "neutral", "contradiction"]
                prediction_dict = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
                max_label = max(prediction_dict, key=prediction_dict.get)
                nli_scores.append(max_label)

            line["nli_scores"] = nli_scores
            if "contradiction" in nli_scores:
                filter_count += 1
                continue
            else:
                self.filtered_generated_instructions.append(line)
            count += 1
            
    def generate_response(self, text, k=2):
        for instruction in self.filtered_generated_instructions:
            prompt = (
                f"Please answer the query strictly following the instruction.\n"
                f"[instruction] {instruction['instruction']}\n"
                f"[Query] {text}"
            )
            
            responses = []
            for _ in range(k):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
                outputs = self.model.generate(input_ids, max_length=1024, do_sample=True, temperature=0.7)
                generated_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                responses.append(generated_text)
            
            self.generated_responses.append({
                "instruction": instruction['instruction'],
                "prompt": prompt,
                "gpt-answer": responses,
                "eval_func": instruction["eval_func"],
            })
                
    def filter_generated_response(self):
        filtered_samples = []
        for result in tqdm(self.generated_responses, desc="Filtering back translated responses"):
            eval_funcs = []
            for func, score in result['eval_func']:
                local_vars = {}
                try:
                    exec(func, globals(), local_vars)
                except Exception as e:
                    print("Error executing eval function:", e)
                    continue
                if 'evaluate' in local_vars:
                    eval_funcs.append(local_vars['evaluate'])
            
            filter_responses = []
            for response in result['gpt-answer']:
                acc = []
                for eval_func in eval_funcs:
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res = eval_func(response)
                    except Exception as e:
                        print(e)
                        res = None
                    finally:
                        signal.alarm(0)
                    if res is not None:
                        try:
                            acc.append(int(res))
                        except Exception:
                            continue
                acc = np.mean(acc) if acc else 0
                if acc > 0:
                    filter_responses.append(response)
            
            for each in filter_responses:
                try:
                    query_match = re.findall(r'\[Query\](.*)$', result['prompt'], re.DOTALL)
                    query = query_match[0].strip() if query_match else ""
                    filtered_samples.append({
                        'instruction': result['instruction'],
                        'query': query,
                        'response': each
                    })
                except IndexError:
                    print("Prompt extraction error:", result['prompt'])
        
        self.filtered_generated_responses = list(map(json.loads, set(map(json.dumps, filtered_samples))))
        
    def filter2_generated_response(self, k=2): 
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
        for each in self.filtered_generated_responses:
            each['prompt'] = prompt_template.format(
                instruction=each['instruction'],
                query=each['query'],
                response=each['response']
            )
            each['gen'] = []
            for _ in range(k):
                input_ids = self.tokenizer.encode(each['prompt'], return_tensors="pt").cuda()
                outputs = self.model.generate(input_ids, max_length=1024, do_sample=True, temperature=0.7)
                generated_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                each['gen'].append(generated_text)
            
            scores = []
            for each in each['gen']:
                score = re.findall(r'Score: (\d+?)$', each)
                if score:
                    scores.append(int(score[0]))
            score = np.mean(scores) if scores else 0
            if score > 8: # quality score
                self.filtered2_generated_responses.append(each)