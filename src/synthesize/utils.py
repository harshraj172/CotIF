import re 
import json
import random
from collections import Counter

import torch

import wn 
import spacy 
from string import punctuation
from spacy.lang.en import stop_words


spacy_nlp = spacy.load('en_core_web_sm')
en_wn = wn.Wordnet('omw-en')

numbering_list = ['3', '7)', '7.', '4', 'iii.', 'iii-', '8.', '4-', 'v:', 'I:', 'ii.', 'i.', 'V)', 'E)', 'I)', 'III.', 'III)', '2-', '1)', 'v-', 'III', 'I.', 'c)', '1.', 'V-', 'iv)', 'A)', 'v)', 'IV', 'C.', 'ii)', 'I', 'IV.', 'C)', 'II-', '2.', 'III-', 'IV)', 'd)', 'iii', 'i-', 'iii:', 'A.', 'B.', '1', '6)', 'ii', '8)', '3)', 'e)', 'ii-', '5-', 'II)', 'iv-', '2)', 'e.', 'IV:', 'III:', 'i)', '10.', 'V', 'V.', 'v.', 'D)', 'E.', 'iv:', 'B)', 'II', 'ii:', 'V:', 'a.', '5.', 'IV-', '9.', 'D.', '3.', '4:', '2:', 'i', 'II.', '3-', '2', 'c.', 'a)', '3:', '10)', 'd.', 'i:', 'iv.', '1-', '4.', '5', 'iv', 'iii)', 'b.', '1:', 'II:', 'v', '5:', '6.', 'b)', 'I-', '9)', '4)', '5)']

stopwords_list = ['es', 'ing', 'ed', 'include', 'includes', 'also', 'haven', 'are', 'why', 'most', "won't", 'against', 'with', 'needn', 'couldn', 'now', 'mustn', 'who', 'under', 'doing', 'am', 'aren', 'they', "didn't", 'd', 'doesn', 'if', 'he', 'her', "haven't", 'isn', 'own', 'does', 'such', 'until', 'into', 'had', 'again', 'over', "hadn't", "you'll", 't', 'by', 'be', "wasn't", 'so', 'yours', 'both', 'any', 'did', "you've", 'these', 'myself', 'o', 'hasn', "isn't", 'you', 'other', 'shan', 'being', 'yourselves', 'was', 'no', 'm', 'those', 'will', 'its', 'itself', 'have', 'down', 'weren', 'having', 'wouldn', 'herself', "mustn't", 'very', 'do', "should've", 'him', "you'd", 'below', 'just', 'that', 'for', 'which', 'but', 'nor', 'all', 'then', 'i', 'whom', 'it', 'once', 'here', 've', "you're", 'ours', "that'll", 'a', 'won', 'himself', 'where', 'this', 'your', "hasn't", 'same', 'when', 'ourselves', 'because', "needn't", 'theirs', 'from', 'mightn', 'my', 'while', 'yourself', "she's", 'each', "doesn't", 'only', 'at', 's', 'their', "wouldn't", 'shouldn', 'and', 'themselves', 'hers', 'has', 'up', 'ma', 'in', 'll', 'we', 're', 'y', 'of', 'after', 'our', "shan't", 'before', 'wasn', 'can', 'should', 'been', 'through', 'as', 'further', 'during', 'between', 'there', 'me', 'on', 'don', "shouldn't", 'more', 'out', "don't", 'the', "weren't", "aren't", "it's", 'what', 'or', "couldn't", 'hadn', "mightn't", 'his', 'above', 'to', 'how', 'few', 'off', 'them', 'didn', 'ain', 'not', 'she', 'an', 'than', 'too', 'is', 'some', 'were', 'about']

common_title_words_set = {'introduction', 'conclusion', 'section', 'chapter', 'works', 'notes', 'note', 'further', 'see', 'references', 'reference', 'section', 'title', 'conclusion', 'intro', 'introduction', 'executive', 'summary', 'key', 'plot', 'theme'}
stopwords_set = set(stopwords_list + numbering_list)


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, filename, append=False):
    mode = "a" if append else "w"  
    with open(filename, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def generate_with_batching(model, tokenizer, batch, use_cache=True, repetition_penalty=1.2, max_new_tokens=400, batch_size=2, skip_special_tokens=True, return_continuations_only=True, group_by_length=20000, **args):

    device = model.device
    output = [None]*len(batch)
    with torch.no_grad():
        lst_by_buckets = {}
        idx_by_buckets = {}
        for idx, a_text in enumerate(batch):
            bucket = int(len(a_text)//group_by_length)
            lst_by_buckets[bucket] = lst_by_buckets.get(bucket, []) + [a_text]
            idx_by_buckets[bucket] = idx_by_buckets.get(bucket, []) + [idx]
        for bucket, batch2 in lst_by_buckets.items():
            idxs = idx_by_buckets[bucket]
            batch_size2 = max(1,int(batch_size/(1+bucket)))
            for rng in range(0, len(batch2), batch_size2):
                sub_batch2 = batch2[rng:min(len(batch2), rng+batch_size2)]
                idxs2 = idxs[rng:min(len(batch2), rng+batch_size2)]
                
                model_inputs = tokenizer(sub_batch2, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, ).to(device)
                prompt_len = model_inputs["input_ids"].shape[-1]
                
                model_output = model.generate(**model_inputs, 
                                                use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens)                    
                if return_continuations_only:
                    model_output = model_output[:, prompt_len:]
                responses = tokenizer.batch_decode(model_output, skip_special_tokens=skip_special_tokens,)
                for idx, r in zip(idxs2, responses):
                    output[idx] = r
                print("responses:", responses)
    output = [text.split("<pad>")[0].split("</s>",1)[0].split("<|im_end|>",1)[0].split("<|endoftext|>",1)[0].rstrip() for text in output]
    return output

def tokenize_with_assistant_continuation(tokenizer, messages):
  """
  Tokenizes chat messages, returning the content up to the assistant's response without any ending tokens.

  This function adapts the tokenization for assistant responses in chat-based templates. It trims any 
  standard ending associated with the assistant's response for continuity in conversations.

  Args:
      tokenizer: Tokenizer instance with support for chat templates and continuation markers.
      messages (list of dict): List of messages in chat format, each having 'role' and 'content' keys.

  Returns:
      str: Tokenized message sequence without the assistant's ending token.
  """
  if not hasattr(tokenizer, "assistant_ending"):
    msg = tokenizer.apply_chat_template([{"role": "user", "content": ""}, {"role": "assistant", "content": "@@@@@@"}], tokenize=False)
    tokenizer.assistant_ending = msg.split("@@@@@@")[-1]
    msg = tokenizer.apply_chat_template([{"role": "user", "content": "!!!!!!!!"}, {"role": "assistant", "content":  "@@@@@@"}, {"role": "user", "content": "<<<<<<<"}], tokenize=False)
    tokenizer.assistant_beginning = msg.split("@@@@@@",1)[0].split("!!!!!!!!",1)[-1]
    user_beginning = msg.split("!!!!!!!!",1)[0]
    user_beginning2 = msg.split( "<<<<<<<",1)[0].split("@@@@@@",1)[-1]
    if len(user_beginning2) < len(user_beginning):
        user_beginning = user_beginning2
    tokenizer.user_beginning  = user_beginning
    tokenizer.user_ending = msg.split( "<<<<<<<",1)[-1]
    
  if not messages: return ""
  return tokenizer.apply_chat_template(messages, tokenize=False)[:-len(tokenizer.assistant_ending)]


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


def check_constraints(text):
    global spacy_nlp, en_wn
    
    def generate_ngrams(tokens, n=4):
        return [tokens[i:i+n] for i in range(len(tokens)-n+1)]

    def count_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return len([s.strip() for s in sentences if s.strip()])
    
    def repeated_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sents = Counter([s.strip() for s in sentences if s.strip()]) 
        return dict([ab for ab in sents.items() if ab[1] > 1])
    
    def palindromes(text):
        words = text.split()
        return [
            word.strip(punctuation)
            for word in words
            if len(word.strip(punctuation)) > 3 and word.strip(punctuation).lower() == word.strip(punctuation).lower()[::-1]
        ]
    def uppercase_word_count(text):
        return sum(1 for word in text.split() if word.isupper())

    def count_ngram_repetitions(text):
        words = text.lower().split()
        ngrams = Counter([" ".join(a) for a in generate_ngrams(words, 5) + \
                         generate_ngrams(words, 6) + \
                          generate_ngrams(words, 7) + \
                          generate_ngrams(words, 8)]) 
                                
        return dict([ab for ab in ngrams.items() if ab[1] > 1])


    def contains_list_or_enumeration(text):
        return (re.match(r"\b(first|second|third|one|two|three|1\.|2\.|3\.)\b", text, re.IGNORECASE))
    
    def analyze_nouns_common_hypernyms(text):
        doc = spacy_nlp(text)
        candidate_nouns = [strip_left_stopwords(e.text)  for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set]        
        noun_synsets = {}
        for noun in candidate_nouns:
            syns = en_wn.synsets(noun, "n")
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
                        if name in {"act", "whole", "communication", "thing", "content", "group", "social group", "unit", "organism", "region", "area", "geographic point", "relation", "attribute", "happening", "action", "entity", "causal agent", 'administrative district', "abstraction", "object", "point", "event", "physical entity", "matter", "part", "person", "state"}:
                            continue
                        hypernym_map.setdefault(name, set()).update([noun_list[i], noun_list[j]])
        
        hypernym_map = {
            hyp: list(nset) for hyp, nset in hypernym_map.items() if len(nset) >= 2
        }

        return hypernym_map

    def count_paragraphs(text):
        paras = [p.strip() for p in text.split('\n') if p.strip()]
        return len(paras)

    def consecutive_alliterative_words(text, n=4):
        words = text.lower().split()
        return any(
            all(words[i + k].startswith(words[i][0]) for k in range(n))
            for i in range(len(words)-n+1)
        )

    def is_haiku(text):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) != 3:
            return False
        # Compute syllable counts using a simple vowel-group method.
        syllable_counts = [
            sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in line.split())
            for line in lines
        ]
        return syllable_counts == [5, 7, 5]

    def hidden_acrostic(text, length=5):
        first_letters = "".join([s.strip()[0].lower() for s in text.split('.') if s.strip()])
        for word in generate_ngrams(first_letters, 6):
            syns = en_wn.synsets(word)
            if syns:
                return word
            
        for word in generate_ngrams(first_letters, 5):
            syns = en_wn.synsets(word)
            if syns:
                return word
            
        for word in generate_ngrams(first_letters, 4):
            syns = en_wn.synsets(word)
            if syns:
                return word            

    def rhyme(text):
        """
        Expects text with 4 lines. Returns True if the first and third lines rhyme
        and the second and fourth lines rhyme (and the two rhymes are different).
        Uses the CMU Pronouncing Dictionary.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) != 4:
            return False
        d = cmudict.dict()
        def get_rhyme(line):
            words_in_line = re.findall(r'\b\w+\b', line)
            if not words_in_line:
                return None
            last_word = words_in_line[-1].lower()
            pronunciations = d.get(last_word)
            if pronunciations:
                # Use the last 3 phonemes as the rhyme signature (if available)
                phonemes = pronunciations[0]
                return tuple(phonemes[-3:]) if len(phonemes) >= 3 else tuple(phonemes)
            return None
        
        rhyme1 = get_rhyme(lines[0])
        rhyme2 = get_rhyme(lines[1])
        rhyme3 = get_rhyme(lines[2])
        rhyme4 = get_rhyme(lines[3])
        return (rhyme1 is not None and rhyme2 is not None and
                rhyme1 == rhyme3 and rhyme2 == rhyme4 and rhyme1 != rhyme2)


    if spacy_nlp is None:
        spacy_nlp = spacy.load('en_core_web_sm')

    if en_wn is None:
        en_wn = wn.Wordnet('omw-en')

    starts_with = {}
    ends_with = {}
    for t in text.split(". "):
        t_arr = t.split(" ")
        if len(t_arr) > 3:
            t2 = " ".join(t_arr[:3]).strip(".?!|;,")
            starts_with[t2] = starts_with.get(t2, 0) + 1
        if len(t_arr) > 4:
            t2 = " ".join(t_arr[:4]).strip(".?!|;,")
            starts_with[t2] = starts_with.get(t2, 0) + 1
        if len(t_arr) > 5:
            t2 = " ".join(t_arr[:5]).strip(".?!|;,")
            starts_with[t2] = starts_with.get(t2, 0) + 1
            
        if len(t_arr) > 5:
            t2 = " ".join(t_arr[-3:]).strip(".?!|;,")
            ends_with[t2] = ends_with.get(t2, 0) + 1
        if len(t_arr) > 6:
            t2 = " ".join(t_arr[-4:]).strip(".?!|;,")
            ends_with[t2] = ends_with.get(t2, 0) + 1
        if len(t_arr) > 7:
            t2 = " ".join(t_arr[-5:]).strip(".?!|;,")
            ends_with[t2] = ends_with.get(t2, 0) + 1
            
    starts_with_para = {}
    ends_with_para = {}
    for t in text.split("\n"):
        t_arr = t.split(" ")
        if len(t_arr) > 3:
            t2 = " ".join(t_arr[:3]).strip(".?!|;,")
            starts_with_para[t2] = starts_with_para.get(t2, 0) + 1
        if len(t_arr) > 4:
            t2 = " ".join(t_arr[:4]).strip(".?!|;,")
            starts_with_para[t2] = starts_with_para.get(t2, 0) + 1
        if len(t_arr) > 5:
            t2 = " ".join(t_arr[:5]).strip(".?!|;,")
            starts_with_para[t2] = starts_with_para.get(t2, 0) + 1
            
        if len(t_arr) > 5:
            t2 = " ".join(t_arr[-3:]).strip(".?!|;,")
            ends_with_para[t2] = ends_with_para.get(t2, 0) + 1
        if len(t_arr) > 6:
            t2 = " ".join(t_arr[-4:]).strip(".?!|;,")
            ends_with_para[t2] = ends_with_para.get(t2, 0) + 1
        if len(t_arr) > 7:
            t2 = " ".join(t_arr[-5:]).strip(".?!|;,")
            ends_with_para[t2] = ends_with_para.get(t2, 0) + 1

            
    for key in list(starts_with_para.keys()):
        if starts_with_para[key] < 2: del starts_with_para[key]
    for key in list(ends_with_para.keys()):
        if ends_with_para[key] < 2: del ends_with_para[key]
        
    for key in list(starts_with.keys()):
        if starts_with[key] < 2: del starts_with[key]
    for key in list(ends_with.keys()):
        if ends_with[key] < 2: del ends_with[key]
        
    results = {
        "palindrome": palindromes(text),        
        "uppercase_word_count": uppercase_word_count(text),
        "ngram_repetitions": count_ngram_repetitions(text),
        "contains_list_or_enumeration": contains_list_or_enumeration(text),
        "common_noun_hypernyms": analyze_nouns_common_hypernyms(text),
        "paragraph_count": count_paragraphs(text),
        "hidden_acrostic": hidden_acrostic(text),
        "para_starts_with": starts_with_para,
        "para_ends_with": ends_with_para,         
        "sent_starts_with": starts_with,
        "sent_ends_with": ends_with,         
        "word_count": len(text.split()),
        "sentence_count": count_sentences(text),
        "repeated_sentences": repeated_sentences(text),        
        "unique_word_count": len(set(text.lower().split())),}

    # # Return a random sample of 5 checks for brevity
    # # Return a random sample of num_returs checks for brevity
    # # You should have more positive checks than negative checks
    # shuff_keys, results = list(checks.keys()), {}
    # random.shuffle(shuff_keys)
    # num_neg = random.randint(0, 1) 
    # num_pos = num_returns - num_neg
    # for key in shuff_keys:
    #     if num_neg > 0 and (not checks[key] or checks[key]==0):
    #         results[key] = checks[key]
    #         num_neg -= 1
    #     if num_pos > 0 and (checks[key] or checks[key] > 0):
    #         results[key] = checks[key]
    #         num_pos -= 1

    return results


constraint_templates = {
    "sentence_count": [
        "Ensure that your response includes exactly {sentence_count} sentences.",
        "Your answer should consist of {sentence_count} sentences.",
        "Craft your reply with precisely {sentence_count} sentences.",
        "Make sure your response contains {sentence_count} sentences.",
        "Your text must be organized into {sentence_count} sentences.",
        "The answer should have {sentence_count} sentences.",
        "Please structure your response to include {sentence_count} sentences.",
        "Include exactly {sentence_count} sentences in your answer.",
        "Format your reply so that it contains {sentence_count} sentences.",
        "The response should comprise {sentence_count} distinct sentences."
    ],
    "unique_word_count": [
        "Ensure that your answer uses {unique_word_count} unique words.",
        "Your response should include {unique_word_count} different words.",
        "Make sure your reply contains {unique_word_count} unique terms.",
        "Craft your answer with a vocabulary of {unique_word_count} distinct words.",
        "The text should incorporate {unique_word_count} unique words.",
        "Please include exactly {unique_word_count} unique words in your answer.",
        "Your answer must feature {unique_word_count} distinct words.",
        "Structure your response to utilize {unique_word_count} unique words.",
        "Ensure your vocabulary consists of {unique_word_count} unique words.",
        "Compose your answer with {unique_word_count} distinct words."
    ],
    "palindrome": [
        "Ensure that your answer contains {palindrome}.",
        "Your response should include {palindrome} as a palindromic words.",
        "Make sure to incorporate {palindrome} palindrome in your reply.",
        "Your text must feature {palindrome} as a palindrome.",
        "Include exactly {palindrome} as a palindrome in your answer.",
        "The answer should have {palindrome_count} element as a palindrome.",
        "Craft your response with {palindrome_count} as a palindromic occurrences.",
        "Please ensure there are {palindrome_count} palindrome in your reply.",
        "Structure your answer to include {palindrome_count} palindromic instances.",
        "Compose your response with {palindrome_count} palindromes."
    ],
    "uppercase_word_count": [
        "Ensure that your answer includes {uppercase_word_count} uppercase words.",
        "Your response should contain {uppercase_word_count} words in uppercase.",
        "Make sure to include {uppercase_word_count} uppercase words in your reply.",
        "Your text must feature {uppercase_word_count} words written in uppercase.",
        "Include exactly {uppercase_word_count} uppercase words in your answer.",
        "Craft your response with {uppercase_word_count} uppercase terms.",
        "Please ensure there are {uppercase_word_count} words in uppercase in your text.",
        "Structure your answer to include {uppercase_word_count} uppercase words.",
        "Compose your response featuring {uppercase_word_count} uppercase words.",
        "The answer should have {uppercase_word_count} uppercase words."
    ],
    "ngram_repetitions": [
        "Ensure that your response contains {ngram_repetitions} repeated n-grams.",
        "Your answer should include {ngram_repetitions} instances of n-gram repetition.",
        "Make sure to incorporate {ngram_repetitions} repeated n-grams in your reply.",
        "Your text must feature {ngram_repetitions} n-gram repetitions.",
        "Include exactly {ngram_repetitions} repeated n-grams in your answer.",
        "Craft your response with {ngram_repetitions} n-gram repetitions.",
        "Please ensure there are {ngram_repetitions} repeated n-grams in your text.",
        "Structure your answer to include {ngram_repetitions} n-gram repetitions.",
        "Compose your response featuring {ngram_repetitions} repeated n-grams.",
        "The reply should have {ngram_repetitions} instances of n-gram repetition."
    ],
    "contains_list_or_enumeration": [
        "Ensure that your response contains {contains_list_or_enumeration} lists or enumerations.",
        "Your answer should include {contains_list_or_enumeration} enumerated lists.",
        "Make sure to incorporate {contains_list_or_enumeration} instances of listing in your reply.",
        "Your text must feature {contains_list_or_enumeration} lists or enumerations.",
        "Include exactly {contains_list_or_enumeration} instances of list or enumeration in your answer.",
        "Craft your response with {contains_list_or_enumeration} lists or enumerations.",
        "Please ensure there are {contains_list_or_enumeration} instances of listing in your text.",
        "Structure your answer to include {contains_list_or_enumeration} lists or enumerations.",
        "Compose your response featuring {contains_list_or_enumeration} lists or enumerations.",
        "The reply should have {contains_list_or_enumeration} instances of enumeration."
    ],
    "common_noun_hypernyms": [
        "Ensure that your response analyzes common noun hypernyms with a count of {common_noun_hypernyms}.",
        "Your answer should include {common_noun_hypernyms} common noun hypernyms.",
        "Make sure to incorporate an analysis of {common_noun_hypernyms} common noun hypernyms in your reply.",
        "Your text must feature an evaluation of {common_noun_hypernyms} common noun hypernyms.",
        "Include exactly {common_noun_hypernyms} common noun hypernyms in your answer.",
        "Craft your response with an assessment of {common_noun_hypernyms} common noun hypernyms.",
        "Please ensure there is an analysis of {common_noun_hypernyms} common noun hypernyms in your text.",
        "Structure your answer to include a review of {common_noun_hypernyms} common noun hypernyms.",
        "Compose your response featuring {common_noun_hypernyms} common noun hypernyms.",
        "The reply should have an analysis of {common_noun_hypernyms} common noun hypernyms."
    ],
    "paragraph_count": [
        "Ensure that your response contains exactly {paragraph_count} paragraphs.",
        "Your answer should consist of {paragraph_count} paragraphs.",
        "Craft your reply with precisely {paragraph_count} paragraphs.",
        "Make sure your response is divided into {paragraph_count} paragraphs.",
        "Your text must be organized into {paragraph_count} paragraphs.",
        "The answer should have {paragraph_count} paragraphs.",
        "Please structure your response to include {paragraph_count} paragraphs.",
        "Include exactly {paragraph_count} paragraphs in your answer.",
        "Format your reply so that it comprises {paragraph_count} paragraphs.",
        "The response should be split into {paragraph_count} distinct paragraphs."
    ],
    "paragraphs_start_with": [
        "Ensure that each paragraph in your response starts with the letter '{random_letter}'.",
        "Your answer should have every paragraph beginning with '{random_letter}'.",
        "Make sure every paragraph in your reply starts with '{random_letter}'.",
        "Your text must feature paragraphs that commence with '{random_letter}'.",
        "Each paragraph in your answer should begin with '{random_letter}'.",
        "Craft your response so that all paragraphs start with '{random_letter}'.",
        "Please ensure that every paragraph starts with the letter '{random_letter}'.",
        "Structure your answer such that each paragraph starts with '{random_letter}'.",
        "Compose your response with paragraphs that all start with '{random_letter}'.",
        "The reply should have each paragraph beginning with '{random_letter}'."
    ],
    "hidden_acrostic": [
        "Ensure that your response contains a hidden acrostic: {hidden_acrostic}.",
        "Your answer should include a concealed acrostic: {hidden_acrostic}.",
        "Make sure to embed a hidden acrostic in your reply: {hidden_acrostic}.",
        "Your text must feature a hidden acrostic: {hidden_acrostic}.",
        "Include a secret acrostic in your answer, marked by {hidden_acrostic}.",
        "Craft your response with a hidden acrostic: {hidden_acrostic}.",
        "Please ensure your reply incorporates a hidden acrostic: {hidden_acrostic}.",
        "Structure your answer to contain a concealed acrostic: {hidden_acrostic}.",
        "Compose your response featuring a hidden acrostic: {hidden_acrostic}.",
        "The reply should have a hidden acrostic element: {hidden_acrostic}."
    ],
    "para_ends_with": [
        "Make sure each paragraph ends with the phrase '{para_ends_with}'.",
        "Conclude every paragraph with '{para_ends_with}'.",
        "Your paragraphs should all finish with the words: '{para_ends_with}'.",
        "Each paragraph must wrap up using the phrase '{para_ends_with}'.",
        "Ensure that the closing line of each paragraph contains '{para_ends_with}'.",
        "Use '{para_ends_with}' to end every paragraph in your response.",
        "Let '{para_ends_with}' serve as the final words in each paragraph.",
        "Every paragraph should end with the exact text: '{para_ends_with}'.",
        "The last phrase of all paragraphs should be '{para_ends_with}'.",
        "Close each paragraph with the following: '{para_ends_with}'."
    ],
    
    "sent_starts_with": [
        "Begin each sentence with the phrase '{sent_starts_with}'.",
        "Make sure every sentence starts with '{sent_starts_with}'.",
        "Your sentences must all open with the words '{sent_starts_with}'.",
        "Use '{sent_starts_with}' to initiate each sentence in your text.",
        "Start every single sentence with the phrase: '{sent_starts_with}'.",
        "Each sentence in your response should begin with '{sent_starts_with}'.",
        "The first few words of each sentence must be '{sent_starts_with}'.",
        "Let '{sent_starts_with}' be the opening of every sentence you write.",
        "Ensure that all your sentences launch with '{sent_starts_with}'.",
        "Begin your sentences consistently with '{sent_starts_with}'."
    ],
    
    "sent_ends_with": [
        "End each sentence with '{sent_ends_with}'.",
        "Your sentences should all finish using the phrase '{sent_ends_with}'.",
        "Conclude every sentence with '{sent_ends_with}' as the last words.",
        "Make sure the final words of each sentence are '{sent_ends_with}'.",
        "Let '{sent_ends_with}' be how every sentence ends.",
        "Each sentence must wrap up with the phrase '{sent_ends_with}'.",
        "Ensure your sentences terminate with '{sent_ends_with}'.",
        "Use '{sent_ends_with}' to end every sentence you write.",
        "Finish your sentences with the exact wording: '{sent_ends_with}'.",
        "Sentences should consistently conclude with '{sent_ends_with}'."
    ],
    
    "word_count": [
        "Make sure your text contains exactly {word_count} words.",
        "Your response should include a total of {word_count} words.",
        "Ensure the word count is precisely {word_count}.",
        "Craft your answer with exactly {word_count} words.",
        "Write a response that adds up to {word_count} words.",
        "Do not exceed or fall short of {word_count} total words.",
        "Use {word_count} words in your reply, no more, no less.",
        "Your answer must contain a word count of {word_count}.",
        "Hit the target of {word_count} words in your text.",
        "Structure your text to be exactly {word_count} words long."
    ],
    
    "sentence_count": [
        "Include exactly {sentence_count} full sentences in your response.",
        "Limit your answer to {sentence_count} sentences only.",
        "Ensure you write precisely {sentence_count} complete sentences.",
        "Stick to {sentence_count} distinct sentences in your reply.",
        "Your text must be composed of {sentence_count} sentences.",
        "Keep your response to a clean {sentence_count} sentences.",
        "Use {sentence_count} sentences to express your answer.",
        "Structure your writing with exactly {sentence_count} sentences.",
        "Craft your response using {sentence_count} standalone sentences.",
        "Write your reply using no more and no fewer than {sentence_count} sentences."
    ],
    
    "repeated_sentences": [
        "Include {repeated_sentences} repeated sentence(s) in your answer.",
        "Your response must reuse {repeated_sentences} sentence(s).",
        "Deliberately repeat a sentence {repeated_sentences} time(s) in your response.",
        "Make sure {repeated_sentences} sentence(s) appear more than once.",
        "Use repetition by including {repeated_sentences} identical sentence(s).",
        "There should be {repeated_sentences} instances of sentence repetition.",
        "Let {repeated_sentences} of your sentences be used more than once.",
        "Repeat a line or phrase exactly {repeated_sentences} time(s).",
        "Echo the same sentence {repeated_sentences} times in your text.",
        "Repetition of sentence(s) should happen {repeated_sentences} time(s) within your response."
    ]
}

NL_MAP = {True: "with",
        False: "without"}
def append_constraints_to_question(question, constraints, templates):
    instructions = []
    for key, value in constraints.items():
        # Check if there are templates for the current constraint.
        if key in templates:
            # Randomly choose one instruction template from the list.
            chosen_template = random.choice(templates[key])
            # Format the chosen template using the constraint value.
            # (For keys with multiple placeholders like {random_letter}, ensure that
            # the 'constraints' dict contains the needed entries.)
            value = NL_MAP[value] if value in list(NL_MAP.keys()) else value
            instruction = chosen_template.format(**{key: value})
            instructions.append(instruction)
        else:
            pass

    appended_question = "Thinking Contraints:\n" + "\n".join(instructions) + f"\n\nQuestion: {question}"
    return appended_question