import os 
import re 
import uuid
import json
import random
from tqdm.auto import tqdm

import nltk
# nltk.data.path.append('./nltk_data')
from nltk.corpus import wordnet
from string import punctuation, ascii_lowercase
from nltk.corpus import stopwords, cmudict, words as nltk_words
from nltk import pos_tag, word_tokenize

import torch


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

def check_constraints(text, num_returns=3):

    def count_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return len([s.strip() for s in sentences if s.strip()])
    
    def count_words_starting_with(text, letter):
        words = text.split()
        return sum(1 for word in words if word.lower().startswith(letter.lower()))
    
    def count_palindromes(text):
        words = text.split()
        return sum(
            1
            for word in words
            if word.strip(punctuation).lower() == word.strip(punctuation).lower()[::-1]
        )
    
    def count_double_letters(text):
        words = text.split()
        return sum(
            1
            for word in words
            if any(word[i] == word[i + 1] for i in range(len(word) - 1))
        )
    
    def has_emojis(text):
        def is_emoji(char):
            emoji_ranges = [
                (0x1F600, 0x1F64F),
                (0x1F300, 0x1F5FF),
                (0x1F680, 0x1F6FF),
                (0x2600, 0x26FF),
                (0x2700, 0x27BF),
                (0xFE00, 0xFE0F),
                (0x1F900, 0x1F9FF),
                (0x1F1E6, 0x1F1FF),
            ]
            return any(start <= ord(char) <= end for start, end in emoji_ranges)
        
        return any(is_emoji(char) for char in text)

    def contains_numbers(text):
        return any(char.isdigit() for char in text)
    
    def contains_punctuation(text):
        return any(char in punctuation for char in text)
    
    def uppercase_word_count(text):
        return sum(1 for word in text.split() if word.isupper())

    def contains_repeated_words(text):
        words = text.lower().split()
        return not all(words[i] != words[i + 1] for i in range(len(words) - 1))
    
    def max_word_repeats(text):
        words = text.lower().split()
        word_counts = {word: words.count(word) for word in set(words)}
        return max(word_counts.values()) if word_counts else 0
    
    def contains_special_characters(text):
        return any(char in """!@#$%^&*()_+=-{}[]|:;"'<>,/""" for char in text)
    
    def starts_with_letter(text, letter):
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        return all(sentence.lower().startswith(letter.lower()) for sentence in sentences if sentence)

    def contains_question(text):
        return '?' in text
    
    def contains_exclamation(text):
        return '!' in text

    def sentence_length_variability(text):
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        lengths = [len(s.split()) for s in sentences]
        return max(lengths) - min(lengths) if lengths else 0

    def count_uncommon_words(text):
        common_words = set(["the", "and", "is", "in", "to", "of", "a", "that", "it", "on", "for", "this", "with"])
        words = text.lower().split()
        return sum(1 for word in words if word not in common_words and len(word) > 5)

    def count_common_word_overuse(text):
        words = text.lower().split()
        common_counts = {word: words.count(word) for word in words if word in ["the", "and", "is", "of", "to", "a"]}
        return max(common_counts.values()) if common_counts else 0

    def count_ngram_repetitions(text, n=3):
        words = text.lower().split()
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        return sum(1 for ngram in set(ngrams) if words.count(ngram) > 1)

    def contains_contradictions(text):
        contradiction_patterns = [
            r"\bbut\b", 
            r"\bhowever\b", 
            r"\bon the other hand\b", 
            r"\byet\b", 
            r"\balthough\b"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in contradiction_patterns)

    def contains_rhetorical_questions(text):
        return bool(re.search(r"\b(isn\'t it|don\'t you think|is that right|right\?)\b", text, re.IGNORECASE))

    def contains_excessive_parentheses(text):
        return text.count("(") + text.count(")") > 4

    def contains_excessive_adverbs(text):
        adverbs = ["really", "very", "actually", "absolutely", "seriously", "extremely", "totally", "literally"]
        return sum(text.lower().count(adverb) for adverb in adverbs) > 5

    def contains_excessive_nominalization(text):
        nominalizations = ["tion", "ment", "ness", "ity", "ism", "ance"]
        words = text.split()
        return sum(1 for word in words if any(word.endswith(suffix) for suffix in nominalizations)) > 5

    def contains_excessive_passive_voice(text):
        passive_patterns = [r"\b(is|was|were|been|being) [a-z]+ed\b"]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in passive_patterns)

    def lacks_pronouns(text):
        pronouns = ["he", "she", "it", "they", "we", "you", "i"]
        return not any(word in text.lower().split() for word in pronouns)

    def contains_hedging_language(text):
        hedging_words = ["might", "maybe", "perhaps", "possibly", "could", "would", "seem"]
        return any(word in text.lower().split() for word in hedging_words)

    def contains_hyperbole(text):
        hyperbole_patterns = [
            r"\balways\b",
            r"\bnever\b",
            r"\bthe best\b",
            r"\bthe worst\b",
            r"\bunbelievable\b",
            r"\bincredible\b",
            r"\babsolutely amazing\b"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in hyperbole_patterns)

    def contains_repetitive_sentence_structures(text):
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        return len(set(len(s.split()) for s in sentences)) < 3

    def contains_list_or_enumeration(text):
        return bool(re.search(r"\b(first|second|third|one|two|three|1\.|2\.|3\.)\b", text, re.IGNORECASE))

    STOPWORDS = set(stopwords.words('english'))

    def count_words_ending_in_x(text):
        words = word_tokenize(text)
        return sum(
            1
            for w in words
            if w.isalpha()
            and w.lower() not in STOPWORDS
            and w.lower().endswith('x')
        )

    def count_words_starting_in_x(text):
        words = word_tokenize(text)
        return sum(
            1
            for w in words
            if w.isalpha()
            and w.lower() not in STOPWORDS
            and w.lower().startswith('x')
        )

    def analyze_nouns_common_hypernyms(text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        candidate_nouns = []
        for (word_, tag_) in tagged:
            if tag_.startswith('NN') and word_.lower() not in STOPWORDS:
                candidate_nouns.append(word_.lower())

        candidate_nouns = list(set(candidate_nouns))
        noun_synsets = {}
        for noun in candidate_nouns:
            syns = wordnet.synsets(noun, pos=wordnet.NOUN)
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
                        if common_hyp.name() != "entity.n.01":
                            hypernym_map.setdefault(common_hyp.name(), set()).update([noun_list[i], noun_list[j]])
        
        hypernym_map = {
            hyp: list(nset) for hyp, nset in hypernym_map.items() if len(nset) >= 2
        }

        return hypernym_map

    def count_paragraphs(text):
        paras = [p.strip() for p in text.split('\n') if p.strip()]
        return len(paras)

    def paragraphs_start_with_letter(text, letter):
        paras = [p.strip() for p in text.split('\n') if p.strip()]
        return all(p.lower().startswith(letter.lower()) for p in paras)

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
        first_letters = [s[0].lower() for s in text.split('.') if s.strip()]
        return any(
            ''.join(first_letters[i:i+length]) in nltk_words.words()
            for i in range(len(first_letters)-length+1)
        )

    def avoids_letter(text, letter='e'):
        return letter not in text.lower()

    def follows_fibonacci_sequence(text):
        word_counts = [len(s.split()) for s in text.split('.') if s.strip()]
        return all(
            word_counts[i+2] == word_counts[i] + word_counts[i+1] 
            for i in range(len(word_counts)-2)
        )

    def color_words_present(text):
        color_map = {
            'a': 'azure', 'b': 'burgundy', 'c': 'cyan', 'd': 'denim',
            'e': 'ebony', 'f': 'fuchsia', 'g': 'green', 'h': 'heliotrope',
            'i': 'indigo', 'j': 'jade', 'k': 'khaki', 'l': 'lavender',
            'm': 'magenta', 'n': 'navy', 'o': 'ochre', 'p': 'purple',
            'q': 'quartz', 'r': 'red', 's': 'scarlet', 't': 'teal',
            'u': 'ultramarine', 'v': 'violet', 'w': 'white', 'x': 'xanthic',
            'y': 'yellow', 'z': 'zaffre'
        }
        first_letters = set(w[0].lower() for w in text.split() if w)
        return all(
            color_map.get(l, '') in text.lower()
            for l in first_letters if l in color_map
        )

    def contains_palindrome_paragraph(text):
        paras = text.split('\n\n')
        return any(
            p.replace(' ','').lower() == p.replace(' ','').lower()[::-1]
            for p in paras if p.strip()
        )

    def abab_rhyme_scheme(text):
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

    def contains_dual_meanings(text):
        return sum(1 for word in text.split() if len(wordnet.synsets(word)) >= 3) > 5

    def mixed_tense_check(text):
        tense_changes = 0
        prev_tense = None
        for word, tag in pos_tag(word_tokenize(text)):
            current_tense = 'past' if tag in ['VBD','VBN'] else 'present'
            if prev_tense and current_tense != prev_tense:
                tense_changes += 1
            prev_tense = current_tense
        return tense_changes > 3

    def contains_html_tags(text):
        html_pattern = re.compile(r'<[^>]+>')
        return bool(html_pattern.search(text))

    def contains_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return bool(url_pattern.search(text))

    def contains_email_addresses(text):
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        return bool(email_pattern.search(text))

    def contains_phone_numbers(text):
        phone_pattern = re.compile(r'\+?\d[\d -]{8,12}\d')
        return bool(phone_pattern.search(text))

    def contains_hashtags(text):
        hashtag_pattern = re.compile(r'#\w+')
        return bool(hashtag_pattern.search(text))

    def contains_mentions(text):
        mention_pattern = re.compile(r'@\w+')
        return bool(mention_pattern.search(text))

    random_letter = random.choice(ascii_lowercase)

    checks = {
        # "word_count": len(text.split()),
        "sentence_count": count_sentences(text),
        "unique_word_count": len(set(text.lower().split())),
        "palindrome_count": count_palindromes(text),
        "double_letter_words": count_double_letters(text),
        "contains_emojis": has_emojis(text),
        "contains_numbers": contains_numbers(text),
        "contains_punctuation": contains_punctuation(text),
        "uppercase_word_count": uppercase_word_count(text),
        "contains_repeated_words": contains_repeated_words(text),
        "max_repeated_word_count": max_word_repeats(text),
        "contains_special_characters": contains_special_characters(text),
        f"starts_with_letter_{random_letter}": starts_with_letter(text, random_letter),
        "contains_question": contains_question(text),
        "contains_exclamation": contains_exclamation(text),
        "sentence_length_variability": sentence_length_variability(text),
        "uncommon_word_usage": count_uncommon_words(text),
        "common_word_overuse": count_common_word_overuse(text),
        "ngram_repetitions": count_ngram_repetitions(text),
        "contains_contradictions": contains_contradictions(text),
        "contains_rhetorical_questions": contains_rhetorical_questions(text),
        "contains_excessive_parentheses": contains_excessive_parentheses(text),
        "contains_excessive_adverbs": contains_excessive_adverbs(text),
        "contains_excessive_passive_voice": contains_excessive_passive_voice(text),
        "contains_hyperbole": contains_hyperbole(text),
        "contains_list_or_enumeration": contains_list_or_enumeration(text),
        "words_ending_in_x": count_words_ending_in_x(text),
        "words_starting_in_x": count_words_starting_in_x(text),
        "common_noun_hypernyms": analyze_nouns_common_hypernyms(text),
        "paragraph_count": count_paragraphs(text),
        f"paragraphs_start_with_{random_letter}": paragraphs_start_with_letter(text, random_letter),
        "consecutive_alliterative_words": consecutive_alliterative_words(text),
        "is_haiku": is_haiku(text),
        "hidden_acrostic": hidden_acrostic(text),
        "avoids_letter_e": avoids_letter(text, 'e'),
        "follows_fibonacci_sequence": follows_fibonacci_sequence(text),
        "color_words_present": color_words_present(text),
        "contains_palindrome_paragraph": contains_palindrome_paragraph(text),
        "abab_rhyme_scheme": abab_rhyme_scheme(text),
        "contains_dual_meanings": contains_dual_meanings(text),
        "mixed_tense_check": mixed_tense_check(text),
        "contains_html_tags": contains_html_tags(text),
        "contains_urls": contains_urls(text),
        "contains_email_addresses": contains_email_addresses(text),
        "contains_phone_numbers": contains_phone_numbers(text),
        "contains_hashtags": contains_hashtags(text),
        "contains_mentions": contains_mentions(text),
    }

    # Return a random sample of 5 checks for brevity
    # Return a random sample of num_returs checks for brevity
    # You should have more positive checks than negative checks
    shuff_keys, results = list(checks.keys()), {}
    random.shuffle(shuff_keys)
    num_neg = random.randint(0, 1) 
    num_pos = num_returns - num_neg
    for key in shuff_keys:
        if num_neg > 0 and (not checks[key] or checks[key]==0):
            results[key] = checks[key]
            num_neg -= 1
        if num_pos > 0 and (checks[key] or checks[key] > 0):
            results[key] = checks[key]
            num_pos -= 1

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
    "palindrome_count": [
        "Ensure that your answer contains {palindrome_count} palindromes.",
        "Your response should include {palindrome_count} palindromic words.",
        "Make sure to incorporate {palindrome_count} palindromes in your reply.",
        "Your text must feature {palindrome_count} instances of palindromes.",
        "Include exactly {palindrome_count} palindromes in your answer.",
        "The answer should have {palindrome_count} palindrome elements.",
        "Craft your response with {palindrome_count} palindromic occurrences.",
        "Please ensure there are {palindrome_count} palindromes in your reply.",
        "Structure your answer to include {palindrome_count} palindromic instances.",
        "Compose your response with {palindrome_count} palindromes."
    ],
    "double_letter_words": [
        "Ensure your response contains {double_letter_words} words with double letters.",
        "Your answer should include {double_letter_words} instances of double-letter words.",
        "Make sure to incorporate {double_letter_words} words featuring double letters in your reply.",
        "Your text must feature {double_letter_words} words that have consecutive identical letters.",
        "Include exactly {double_letter_words} double-letter words in your answer.",
        "The reply should contain {double_letter_words} words with repeated letters.",
        "Craft your response using {double_letter_words} words that include double letters.",
        "Please ensure there are {double_letter_words} words with double letters in your text.",
        "Structure your answer to comprise {double_letter_words} double-letter words.",
        "Compose your response with {double_letter_words} words containing double letters."
    ],
    "contains_emojis": [
        "Ensure that your response contains {contains_emojis} emojis.",
        "Your answer should include {contains_emojis} emojis.",
        "Make sure to incorporate {contains_emojis} emojis in your reply.",
        "Your text must feature {contains_emojis} emojis.",
        "Include exactly {contains_emojis} emojis in your answer.",
        "Craft your response with {contains_emojis} emojis.",
        "Please ensure there are {contains_emojis} emojis in your text.",
        "Structure your answer to include {contains_emojis} emojis.",
        "Compose your response featuring {contains_emojis} emojis.",
        "The reply should have {contains_emojis} emojis."
    ],
    "contains_numbers": [
        "Ensure that your response includes {contains_numbers} numerical figures.",
        "Your answer should incorporate {contains_numbers} numbers.",
        "Make sure to include {contains_numbers} numeric characters in your reply.",
        "Your text must feature {contains_numbers} numbers.",
        "Include exactly {contains_numbers} numbers in your answer.",
        "Craft your response with {contains_numbers} numerical elements.",
        "Please ensure there are {contains_numbers} numbers present in your text.",
        "Structure your answer to comprise {contains_numbers} numeric characters.",
        "Compose your response with {contains_numbers} numbers.",
        "The reply should contain {contains_numbers} numerical figures."
    ],
    "contains_punctuation": [
        "Ensure that your response contains {contains_punctuation} punctuation marks.",
        "Your answer should include {contains_punctuation} punctuation symbols.",
        "Make sure to incorporate {contains_punctuation} punctuation elements in your reply.",
        "Your text must feature {contains_punctuation} punctuation marks.",
        "Include exactly {contains_punctuation} punctuation symbols in your answer.",
        "Craft your response with {contains_punctuation} punctuation characters.",
        "Please ensure there are {contains_punctuation} punctuation marks in your text.",
        "Structure your answer to include {contains_punctuation} punctuation marks.",
        "Compose your response featuring {contains_punctuation} punctuation elements.",
        "The reply should have {contains_punctuation} punctuation symbols."
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
    "contains_repeated_words": [
        "Ensure that your response includes repeated words as required: {contains_repeated_words}.",
        "Your answer should demonstrate repeated words with a count of {contains_repeated_words}.",
        "Make sure your reply contains repeated words, totaling {contains_repeated_words}.",
        "Your text must feature {contains_repeated_words} instances of repeated words.",
        "Include exactly {contains_repeated_words} occurrences of repeated words in your answer.",
        "Craft your response so that it has {contains_repeated_words} repeated words.",
        "Please ensure there are {contains_repeated_words} repeated words in your reply.",
        "Structure your answer to comprise {contains_repeated_words} repeated words.",
        "Compose your response with {contains_repeated_words} repeated words.",
        "The answer should have {contains_repeated_words} instances of repeated words."
    ],
    "max_repeated_word_count": [
        "Ensure that the maximum repetition of any word in your answer is {max_repeated_word_count}.",
        "Your response should not exceed {max_repeated_word_count} repetitions for any word.",
        "Make sure no word in your reply is repeated more than {max_repeated_word_count} times.",
        "The answer should have a maximum of {max_repeated_word_count} repeats for any single word.",
        "Include constraints such that no word repeats more than {max_repeated_word_count} times.",
        "Craft your response with a cap of {max_repeated_word_count} on repeated words.",
        "Please ensure that word repetition does not exceed {max_repeated_word_count} in your answer.",
        "Structure your text so that the maximum repeated word count is {max_repeated_word_count}.",
        "Compose your reply ensuring no word appears more than {max_repeated_word_count} times.",
        "The reply should have a limit of {max_repeated_word_count} for any repeated word."
    ],
    "contains_special_characters": [
        "Ensure that your response contains {contains_special_characters} special characters.",
        "Your answer should include {contains_special_characters} special symbols.",
        "Make sure to incorporate {contains_special_characters} special characters in your reply.",
        "Your text must feature {contains_special_characters} special characters.",
        "Include exactly {contains_special_characters} special characters in your answer.",
        "Craft your response with {contains_special_characters} special symbols.",
        "Please ensure there are {contains_special_characters} special characters in your text.",
        "Structure your answer to include {contains_special_characters} special characters.",
        "Compose your response featuring {contains_special_characters} special symbols.",
        "The reply should have {contains_special_characters} special characters."
    ],
    # For constraints that depend on a random letter, we use a placeholder {random_letter}
    "starts_with_letter": [
        "Ensure that your response starts with the letter '{random_letter}'.",
        "Your answer should begin with the letter '{random_letter}'.",
        "Make sure your reply starts with '{random_letter}'.",
        "Your text must commence with the letter '{random_letter}'.",
        "Include an opening letter '{random_letter}' at the start of your answer.",
        "Craft your response so that it starts with '{random_letter}'.",
        "Please begin your reply with the letter '{random_letter}'.",
        "Structure your answer to start with '{random_letter}'.",
        "Compose your response beginning with '{random_letter}'.",
        "The reply should initiate with the letter '{random_letter}'."
    ],
    "contains_question": [
        "Ensure that your response contains {contains_question} question marks.",
        "Your answer should include {contains_question} questions.",
        "Make sure to incorporate {contains_question} interrogative sentences in your reply.",
        "Your text must feature {contains_question} question marks.",
        "Include exactly {contains_question} questions in your answer.",
        "Craft your response with {contains_question} interrogative elements.",
        "Please ensure there are {contains_question} questions in your text.",
        "Structure your answer to include {contains_question} question marks.",
        "Compose your response featuring {contains_question} questions.",
        "The reply should have {contains_question} question marks."
    ],
    "contains_exclamation": [
        "Ensure that your response contains {contains_exclamation} exclamation marks.",
        "Your answer should include {contains_exclamation} exclamations.",
        "Make sure to incorporate {contains_exclamation} exclamatory sentences in your reply.",
        "Your text must feature {contains_exclamation} exclamation marks.",
        "Include exactly {contains_exclamation} exclamations in your answer.",
        "Craft your response with {contains_exclamation} exclamatory elements.",
        "Please ensure there are {contains_exclamation} exclamation marks in your text.",
        "Structure your answer to include {contains_exclamation} exclamations.",
        "Compose your response featuring {contains_exclamation} exclamation marks.",
        "The reply should have {contains_exclamation} exclamation points."
    ],
    "sentence_length_variability": [
        "Ensure that your response exhibits a sentence length variability of {sentence_length_variability}.",
        "Your answer should demonstrate a sentence length variability of {sentence_length_variability}.",
        "Make sure your reply has a sentence length variation of {sentence_length_variability}.",
        "Your text must reflect a sentence length variability of {sentence_length_variability}.",
        "Include a sentence length variability of {sentence_length_variability} in your answer.",
        "Craft your response with a sentence length variability of {sentence_length_variability}.",
        "Please ensure your reply shows a variability in sentence lengths of {sentence_length_variability}.",
        "Structure your answer to feature a sentence length variability of {sentence_length_variability}.",
        "Compose your response with a sentence length variation of {sentence_length_variability}.",
        "The reply should have a sentence length variability of {sentence_length_variability}."
    ],
    "uncommon_word_usage": [
        "Ensure that your answer includes {uncommon_word_usage} uncommon words.",
        "Your response should feature {uncommon_word_usage} rare vocabulary terms.",
        "Make sure to incorporate {uncommon_word_usage} uncommon words in your reply.",
        "Your text must utilize {uncommon_word_usage} unique words.",
        "Include exactly {uncommon_word_usage} uncommon words in your answer.",
        "Craft your response with {uncommon_word_usage} rare words.",
        "Please ensure there are {uncommon_word_usage} uncommon terms in your text.",
        "Structure your answer to include {uncommon_word_usage} rare words.",
        "Compose your response featuring {uncommon_word_usage} uncommon vocabulary.",
        "The reply should have {uncommon_word_usage} uncommon words."
    ],
    "common_word_overuse": [
        "Ensure that your answer does not overuse common words beyond {common_word_overuse} times.",
        "Your response should limit common word usage to {common_word_overuse} occurrences.",
        "Make sure that no common word is repeated more than {common_word_overuse} times in your reply.",
        "Your text must restrict common word overuse to {common_word_overuse} instances.",
        "Limit the repetition of common words to {common_word_overuse} times in your answer.",
        "Craft your response so that common words are used no more than {common_word_overuse} times.",
        "Please ensure that common words are not overused more than {common_word_overuse} occurrences in your text.",
        "Structure your answer with a maximum common word usage of {common_word_overuse}.",
        "Compose your reply limiting common word overuse to {common_word_overuse} times.",
        "The reply should adhere to a common word repetition limit of {common_word_overuse}."
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
    "contains_contradictions": [
        "Ensure that your response highlights {contains_contradictions} contradictions.",
        "Your answer should include {contains_contradictions} contradictory elements.",
        "Make sure to incorporate {contains_contradictions} instances of contradiction in your reply.",
        "Your text must feature {contains_contradictions} contradictions.",
        "Include exactly {contains_contradictions} contradictions in your answer.",
        "Craft your response with {contains_contradictions} contradictory statements.",
        "Please ensure there are {contains_contradictions} contradictions in your text.",
        "Structure your answer to include {contains_contradictions} contradictory elements.",
        "Compose your response featuring {contains_contradictions} contradictions.",
        "The reply should have {contains_contradictions} instances of contradiction."
    ],
    "contains_rhetorical_questions": [
        "Ensure that your response contains {contains_rhetorical_questions} rhetorical questions.",
        "Your answer should include {contains_rhetorical_questions} instances of rhetorical questioning.",
        "Make sure to incorporate {contains_rhetorical_questions} rhetorical questions in your reply.",
        "Your text must feature {contains_rhetorical_questions} rhetorical questions.",
        "Include exactly {contains_rhetorical_questions} rhetorical questions in your answer.",
        "Craft your response with {contains_rhetorical_questions} rhetorical questions.",
        "Please ensure there are {contains_rhetorical_questions} rhetorical questions in your text.",
        "Structure your answer to include {contains_rhetorical_questions} rhetorical questions.",
        "Compose your response featuring {contains_rhetorical_questions} rhetorical questions.",
        "The reply should have {contains_rhetorical_questions} instances of rhetorical questions."
    ],
    "contains_excessive_parentheses": [
        "Ensure that your response contains {contains_excessive_parentheses} instances of excessive parentheses.",
        "Your answer should include {contains_excessive_parentheses} excessive parentheses.",
        "Make sure to incorporate {contains_excessive_parentheses} sets of excessive parentheses in your reply.",
        "Your text must feature {contains_excessive_parentheses} instances of overused parentheses.",
        "Include exactly {contains_excessive_parentheses} occurrences of excessive parentheses in your answer.",
        "Craft your response with {contains_excessive_parentheses} instances of unnecessary parentheses.",
        "Please ensure there are {contains_excessive_parentheses} excessive parentheses in your text.",
        "Structure your answer to include {contains_excessive_parentheses} overused parentheses.",
        "Compose your response featuring {contains_excessive_parentheses} excessive parentheses.",
        "The reply should have {contains_excessive_parentheses} instances of redundant parentheses."
    ],
    "contains_excessive_adverbs": [
        "Ensure that your response contains {contains_excessive_adverbs} instances of excessive adverbs.",
        "Your answer should include {contains_excessive_adverbs} overused adverbs.",
        "Make sure to incorporate {contains_excessive_adverbs} adverbs in your reply.",
        "Your text must feature {contains_excessive_adverbs} instances of adverbial excess.",
        "Include exactly {contains_excessive_adverbs} excessive adverbs in your answer.",
        "Craft your response with {contains_excessive_adverbs} adverbs.",
        "Please ensure there are {contains_excessive_adverbs} excessive adverbs in your text.",
        "Structure your answer to include {contains_excessive_adverbs} overused adverbs.",
        "Compose your response featuring {contains_excessive_adverbs} adverbs.",
        "The reply should have {contains_excessive_adverbs} instances of adverbial overuse."
    ],
    "contains_excessive_passive_voice": [
        "Ensure that your response contains {contains_excessive_passive_voice} instances of passive voice.",
        "Your answer should include {contains_excessive_passive_voice} passive voice constructions.",
        "Make sure to incorporate {contains_excessive_passive_voice} instances of passive voice in your reply.",
        "Your text must feature {contains_excessive_passive_voice} instances of passive voice usage.",
        "Include exactly {contains_excessive_passive_voice} passive voice constructions in your answer.",
        "Craft your response with {contains_excessive_passive_voice} instances of passive voice.",
        "Please ensure there are {contains_excessive_passive_voice} instances of passive voice in your text.",
        "Structure your answer to include {contains_excessive_passive_voice} passive voice constructions.",
        "Compose your response featuring {contains_excessive_passive_voice} instances of passive voice.",
        "The reply should have {contains_excessive_passive_voice} passive voice instances."
    ],
    "contains_hyperbole": [
        "Ensure that your response contains {contains_hyperbole} instances of hyperbole.",
        "Your answer should include {contains_hyperbole} hyperbolic expressions.",
        "Make sure to incorporate {contains_hyperbole} hyperbolic statements in your reply.",
        "Your text must feature {contains_hyperbole} instances of exaggeration.",
        "Include exactly {contains_hyperbole} hyperbolic elements in your answer.",
        "Craft your response with {contains_hyperbole} instances of hyperbole.",
        "Please ensure there are {contains_hyperbole} hyperbolic expressions in your text.",
        "Structure your answer to include {contains_hyperbole} exaggerations.",
        "Compose your response featuring {contains_hyperbole} hyperbolic phrases.",
        "The reply should have {contains_hyperbole} instances of hyperbole."
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
    "words_ending_in_x": [
        "Ensure that your response contains {words_ending_in_x} words ending with the letter 'x'.",
        "Your answer should include {words_ending_in_x} words that end in 'x'.",
        "Make sure to incorporate {words_ending_in_x} words ending in 'x' in your reply.",
        "Your text must feature {words_ending_in_x} words ending with 'x'.",
        "Include exactly {words_ending_in_x} words that finish with the letter 'x' in your answer.",
        "Craft your response with {words_ending_in_x} words ending in 'x'.",
        "Please ensure there are {words_ending_in_x} words that end in 'x' in your text.",
        "Structure your answer to include {words_ending_in_x} words ending in 'x'.",
        "Compose your response featuring {words_ending_in_x} words that conclude with 'x'.",
        "The reply should have {words_ending_in_x} words ending with the letter 'x'."
    ],
    "words_starting_in_x": [
        "Ensure that your response contains {words_starting_in_x} words starting with the letter 'x'.",
        "Your answer should include {words_starting_in_x} words that begin with 'x'.",
        "Make sure to incorporate {words_starting_in_x} words starting with 'x' in your reply.",
        "Your text must feature {words_starting_in_x} words beginning with the letter 'x'.",
        "Include exactly {words_starting_in_x} words that start with 'x' in your answer.",
        "Craft your response with {words_starting_in_x} words that commence with 'x'.",
        "Please ensure there are {words_starting_in_x} words starting with 'x' in your text.",
        "Structure your answer to include {words_starting_in_x} words that begin with 'x'.",
        "Compose your response featuring {words_starting_in_x} words starting with 'x'.",
        "The reply should have {words_starting_in_x} words beginning with 'x'."
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
    "consecutive_alliterative_words": [
        "Ensure that your response contains {consecutive_alliterative_words} instances of consecutive alliterative words.",
        "Your answer should include {consecutive_alliterative_words} groups of alliterative words in succession.",
        "Make sure to incorporate {consecutive_alliterative_words} consecutive alliterative words in your reply.",
        "Your text must feature {consecutive_alliterative_words} instances of alliteration in consecutive words.",
        "Include exactly {consecutive_alliterative_words} occurrences of consecutive alliterative words in your answer.",
        "Craft your response with {consecutive_alliterative_words} instances of consecutive alliterative terms.",
        "Please ensure there are {consecutive_alliterative_words} instances of alliterative word sequences in your text.",
        "Structure your answer to include {consecutive_alliterative_words} consecutive alliterative words.",
        "Compose your response featuring {consecutive_alliterative_words} alliterative word groups.",
        "The reply should have {consecutive_alliterative_words} instances of consecutive alliterative words."
    ],
    "is_haiku": [
        "Ensure that your response is structured as a haiku: {is_haiku}.",
        "Your answer should follow the haiku format: {is_haiku}.",
        "Make sure your reply adheres to the haiku structure: {is_haiku}.",
        "Your text must be composed as a haiku: {is_haiku}.",
        "Include a haiku format in your answer, indicated by {is_haiku}.",
        "Craft your response as a haiku if applicable: {is_haiku}.",
        "Please ensure your reply follows the haiku style: {is_haiku}.",
        "Structure your answer to conform to the haiku format: {is_haiku}.",
        "Compose your response in the form of a haiku: {is_haiku}.",
        "The reply should be arranged as a haiku: {is_haiku}."
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
    "avoids_letter_e": [
        "Ensure that your response avoids using the letter 'e', as required: {avoids_letter_e}.",
        "Your answer should be crafted to avoid the letter 'e': {avoids_letter_e}.",
        "Make sure your reply does not include the letter 'e': {avoids_letter_e}.",
        "Your text must refrain from using the letter 'e': {avoids_letter_e}.",
        "Compose your answer while avoiding the letter 'e': {avoids_letter_e}.",
        "Craft your response to exclude the letter 'e': {avoids_letter_e}.",
        "Please ensure your reply avoids any usage of the letter 'e': {avoids_letter_e}.",
        "Structure your answer so that it does not contain the letter 'e': {avoids_letter_e}.",
        "Your response should completely omit the letter 'e': {avoids_letter_e}.",
        "The reply must be free of the letter 'e': {avoids_letter_e}."
    ],
    "follows_fibonacci_sequence": [
        "Ensure that your response follows a Fibonacci sequence pattern: {follows_fibonacci_sequence}.",
        "Your answer should incorporate elements that follow the Fibonacci sequence: {follows_fibonacci_sequence}.",
        "Make sure your reply adheres to a Fibonacci sequence: {follows_fibonacci_sequence}.",
        "Your text must be structured to follow the Fibonacci sequence: {follows_fibonacci_sequence}.",
        "Include a Fibonacci sequence in your answer, as indicated by {follows_fibonacci_sequence}.",
        "Craft your response with a Fibonacci sequence pattern: {follows_fibonacci_sequence}.",
        "Please ensure your reply follows a Fibonacci sequence: {follows_fibonacci_sequence}.",
        "Structure your answer to reflect a Fibonacci sequence: {follows_fibonacci_sequence}.",
        "Compose your response featuring a Fibonacci sequence: {follows_fibonacci_sequence}.",
        "The reply should have a structure that follows the Fibonacci sequence: {follows_fibonacci_sequence}."
    ],
    "color_words_present": [
        "Ensure that your response contains {color_words_present} color words.",
        "Your answer should include {color_words_present} words that denote colors.",
        "Make sure to incorporate {color_words_present} color-related words in your reply.",
        "Your text must feature {color_words_present} color words.",
        "Include exactly {color_words_present} color words in your answer.",
        "Craft your response with {color_words_present} words associated with colors.",
        "Please ensure there are {color_words_present} color words in your text.",
        "Structure your answer to include {color_words_present} color words.",
        "Compose your response featuring {color_words_present} words that represent colors.",
        "The reply should have {color_words_present} color words."
    ],
    "contains_palindrome_paragraph": [
        "Ensure that your response contains a paragraph with a palindrome: {contains_palindrome_paragraph}.",
        "Your answer should include a paragraph featuring a palindrome: {contains_palindrome_paragraph}.",
        "Make sure to incorporate a palindrome paragraph in your reply: {contains_palindrome_paragraph}.",
        "Your text must feature a paragraph that is a palindrome: {contains_palindrome_paragraph}.",
        "Include exactly {contains_palindrome_paragraph} palindrome paragraph in your answer.",
        "Craft your response with a palindrome paragraph: {contains_palindrome_paragraph}.",
        "Please ensure there is a paragraph containing a palindrome in your text: {contains_palindrome_paragraph}.",
        "Structure your answer to include a paragraph that is a palindrome: {contains_palindrome_paragraph}.",
        "Compose your response featuring a palindrome paragraph: {contains_palindrome_paragraph}.",
        "The reply should have a paragraph with a palindrome: {contains_palindrome_paragraph}."
    ],
    "abab_rhyme_scheme": [
        "Ensure that your response follows an ABAB rhyme scheme: {abab_rhyme_scheme}.",
        "Your answer should adhere to an ABAB rhyme pattern: {abab_rhyme_scheme}.",
        "Make sure your reply follows the ABAB rhyme scheme: {abab_rhyme_scheme}.",
        "Your text must be structured with an ABAB rhyme scheme: {abab_rhyme_scheme}.",
        "Include an ABAB rhyme pattern in your answer, as indicated by {abab_rhyme_scheme}.",
        "Craft your response to follow an ABAB rhyme scheme: {abab_rhyme_scheme}.",
        "Please ensure your reply adheres to the ABAB rhyme pattern: {abab_rhyme_scheme}.",
        "Structure your answer with an ABAB rhyme scheme: {abab_rhyme_scheme}.",
        "Compose your response featuring an ABAB rhyme scheme: {abab_rhyme_scheme}.",
        "The reply should have an ABAB rhyme pattern: {abab_rhyme_scheme}."
    ],
    "contains_dual_meanings": [
        "Ensure that your response includes {contains_dual_meanings} instances of dual meanings.",
        "Your answer should feature {contains_dual_meanings} elements with dual meanings.",
        "Make sure to incorporate {contains_dual_meanings} ambiguous phrases in your reply.",
        "Your text must contain {contains_dual_meanings} instances of dual meanings.",
        "Include exactly {contains_dual_meanings} dual-meaning elements in your answer.",
        "Craft your response with {contains_dual_meanings} instances of double entendre.",
        "Please ensure there are {contains_dual_meanings} instances of dual meanings in your text.",
        "Structure your answer to include {contains_dual_meanings} dual-meaning phrases.",
        "Compose your response featuring {contains_dual_meanings} ambiguous elements.",
        "The reply should have {contains_dual_meanings} instances of dual meanings."
    ],
    "mixed_tense_check": [
        "Ensure that your response maintains a consistent tense, avoiding mixed tenses as indicated by {mixed_tense_check}.",
        "Your answer should adhere to a single tense, with mixed tenses not exceeding {mixed_tense_check}.",
        "Make sure your reply does not mix tenses, as constrained by {mixed_tense_check}.",
        "Your text must be consistent in tense, with mixed tenses limited to {mixed_tense_check} instances.",
        "Include a tense consistency in your answer, keeping mixed tenses to {mixed_tense_check}.",
        "Craft your response with a focus on a single tense, allowing only {mixed_tense_check} mixed instances.",
        "Please ensure there are no more than {mixed_tense_check} instances of mixed tenses in your text.",
        "Structure your answer to maintain tense consistency, with mixed tenses capped at {mixed_tense_check}.",
        "Compose your response in a consistent tense, limiting mixed tenses to {mixed_tense_check}.",
        "The reply should exhibit tense consistency, with only {mixed_tense_check} mixed tense occurrences."
    ],
    "contains_html_tags": [
        "Ensure that your response contains {contains_html_tags} HTML tags.",
        "Your answer should include {contains_html_tags} HTML elements.",
        "Make sure to incorporate {contains_html_tags} HTML tags in your reply.",
        "Your text must feature {contains_html_tags} HTML tags.",
        "Include exactly {contains_html_tags} HTML tags in your answer.",
        "Craft your response with {contains_html_tags} HTML elements.",
        "Please ensure there are {contains_html_tags} HTML tags in your text.",
        "Structure your answer to include {contains_html_tags} HTML tags.",
        "Compose your response featuring {contains_html_tags} HTML tags.",
        "The reply should have {contains_html_tags} HTML tags."
    ],
    "contains_urls": [
        "Ensure that your response contains {contains_urls} URLs.",
        "Your answer should include {contains_urls} web links.",
        "Make sure to incorporate {contains_urls} URLs in your reply.",
        "Your text must feature {contains_urls} URLs.",
        "Include exactly {contains_urls} URLs in your answer.",
        "Craft your response with {contains_urls} web addresses.",
        "Please ensure there are {contains_urls} URLs in your text.",
        "Structure your answer to include {contains_urls} URLs.",
        "Compose your response featuring {contains_urls} URLs.",
        "The reply should have {contains_urls} URLs."
    ],
    "contains_email_addresses": [
        "Ensure that your response contains {contains_email_addresses} email addresses.",
        "Your answer should include {contains_email_addresses} email addresses.",
        "Make sure to incorporate {contains_email_addresses} email addresses in your reply.",
        "Your text must feature {contains_email_addresses} email addresses.",
        "Include exactly {contains_email_addresses} email addresses in your answer.",
        "Craft your response with {contains_email_addresses} email addresses.",
        "Please ensure there are {contains_email_addresses} email addresses in your text.",
        "Structure your answer to include {contains_email_addresses} email addresses.",
        "Compose your response featuring {contains_email_addresses} email addresses.",
        "The reply should have {contains_email_addresses} email addresses."
    ],
    "contains_phone_numbers": [
        "Ensure that your response contains {contains_phone_numbers} phone numbers.",
        "Your answer should include {contains_phone_numbers} contact numbers.",
        "Make sure to incorporate {contains_phone_numbers} phone numbers in your reply.",
        "Your text must feature {contains_phone_numbers} phone numbers.",
        "Include exactly {contains_phone_numbers} phone numbers in your answer.",
        "Craft your response with {contains_phone_numbers} phone numbers.",
        "Please ensure there are {contains_phone_numbers} phone numbers in your text.",
        "Structure your answer to include {contains_phone_numbers} phone numbers.",
        "Compose your response featuring {contains_phone_numbers} phone numbers.",
        "The reply should have {contains_phone_numbers} phone numbers."
    ],
    "contains_hashtags": [
        "Ensure that your response contains {contains_hashtags} hashtags.",
        "Your answer should include {contains_hashtags} hashtag(s).",
        "Make sure to incorporate {contains_hashtags} hashtags in your reply.",
        "Your text must feature {contains_hashtags} hashtags.",
        "Include exactly {contains_hashtags} hashtags in your answer.",
        "Craft your response with {contains_hashtags} hashtags.",
        "Please ensure there are {contains_hashtags} hashtags in your text.",
        "Structure your answer to include {contains_hashtags} hashtags.",
        "Compose your response featuring {contains_hashtags} hashtags.",
        "The reply should have {contains_hashtags} hashtag(s)."
    ],
    "contains_mentions": [
        "Ensure that your response contains {contains_mentions} mentions.",
        "Your answer should include {contains_mentions} mentions of users or entities.",
        "Make sure to incorporate {contains_mentions} mentions in your reply.",
        "Your text must feature {contains_mentions} mentions.",
        "Include exactly {contains_mentions} mentions in your answer.",
        "Craft your response with {contains_mentions} user or entity mentions.",
        "Please ensure there are {contains_mentions} mentions in your text.",
        "Structure your answer to include {contains_mentions} mentions.",
        "Compose your response featuring {contains_mentions} mentions.",
        "The reply should have {contains_mentions} mentions."
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