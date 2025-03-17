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

def check_constraints(text):

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
        "word_count": len(text.split()),
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
    selected_keys = random.sample(list(checks.keys()), 5)
    results = {key: checks[key] for key in selected_keys}

    return results