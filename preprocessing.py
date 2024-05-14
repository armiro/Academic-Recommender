"""
preprocessing functions for input dataset
"""
import re
import logging
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import nltk

nltk.download('stopwords', quiet=True)
STOP_WORDS = stopwords.words('english') + ['…', '&', '...']


def str_to_list(input_str):
    """
    use regex to split string into list of substrings

    :param input_str: string
    :return: list
    """
    delimiters = r', |; |\| | - '
    return list(re.split(pattern=delimiters, string=input_str))


def normalize(input_list):
    """
    make all words lower and clean indents

    :param input_list: list of strings
    :return: list
    """
    return [e.lower().strip() for e in input_list]


def remove_stop_words_from(input_list, method, stop_words):
    """
    remove stop words like 'of' or 'and'

    :param input_list: list of strings
    :param method: string, splitting method to tokenize strings; ['phrase' or 'token']
    :param stop_words: list of strings, stop words & tokens
    :return: list
    """
    # TODO: remove empty strings
    if method == 'token':
        return [e for w in input_list for e in w.split() if e not in stop_words]  # & return tokens
    elif method == 'phrase':
        return [' '.join([e for e in w.split() if e not in stop_words]) for w in input_list]
    else:
        raise ValueError("splitting method should either be 'phrase' or 'token'")


# TODO: use local vocab to make the dictionary more robust
def check_spells_in(input_list):
    """
    check for any possible typos using spell checker

    :param input_list: list of strings
    :return: none
    """
    speller = SpellChecker(language='en', distance=2, case_sensitive=False)
    for phrase in input_list:
        misspelled = speller.unknown(phrase.split())
        if misspelled:
            logging.critical('unknown word: %s', misspelled)
            logging.critical('best correction: %s', speller.correction(misspelled.pop()))


def split_into_tokens(input_list):
    """
    convert list of interests into single words

    :param input_list: list of strings
    :return: list
    """
    return [token for word in input_list for token in word.split()]


# TODO: remove punctuations, numbers, links
def preprocess(input_str, method):
    """
    apply preprocessing steps on the input string

    :param input_str: string
    :param method: string, splitting method to tokenize strings; ['phrase' or 'token']
    :return: list
    """
    input_list = str_to_list(input_str)
    normalized_list = normalize(input_list)
    cleaned_list = remove_stop_words_from(normalized_list, method, stop_words=STOP_WORDS)
    # check_spells_in(cleaned_list)  # view-only; no output yet
    # extracted_tokens_list = split_into_tokens(cleaned_list)
    return cleaned_list
