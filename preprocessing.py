import re
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import nltk

nltk.download('stopwords', quiet=True)


def str_to_list(input_str):
    delimiters = ', |; |-'
    return [e for e in re.split(pattern=delimiters, string=input_str)]


def normalize(input_list):  # make all words lower and clean indents
    return [e.lower().strip() for e in input_list]


def remove_stop_words_from(input_list, method):  # remove stop words like 'of', 'and'
    stop_words = stopwords.words('english')
    if method == 'tokenize':
        return [e for w in input_list for e in w.split() if e not in stop_words]  # & return tokens
    elif method == 'phrase':
        return [' '.join([e for e in w.split() if e not in stop_words]) for w in input_list]
    else:
        raise ValueError("splitting method should either be 'phrase' or 'tokenize'")


# TODO: use local vocab to make the dictionary more robust
def check_spells_in(input_list):  # check for any typos
    speller = SpellChecker(language='en', distance=2, case_sensitive=False)
    for phrase in input_list:
        misspelled = speller.unknown(phrase.split())
        if misspelled:
            print('unknown word:', misspelled)
            print('best correction:', speller.correction(misspelled.pop()))


def split_into_tokens(input_list):  # convert list of interests into single words
    return [token for word in input_list for token in word.split()]


# TODO: remove punctuations, numbers, links
def preprocess(input_str, method):
    input_list = str_to_list(input_str)
    normalized_list = normalize(input_list)
    cleaned_list = remove_stop_words_from(normalized_list, method)
    # check_spells_in(cleaned_list)  # view-only; no output yet
    # extracted_tokens_list = split_into_tokens(cleaned_list)
    return cleaned_list
