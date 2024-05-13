"""
preprocessing functions for input dataset
"""
import re
import logging
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import nltk

nltk.download('stopwords', quiet=True)


def str_to_list(input_str):
    """
    use regex to split string into list of substrings

    :param input_str: string
    :return: list
    """
    delimiters = ', |; |\| | - '
    return list(re.split(pattern=delimiters, string=input_str))


def normalize(input_list):
    """
    make all words lower and clean indents

    :param input_list: list of strings
    :return: list
    """
    return [e.lower().strip() for e in input_list]


def remove_stop_words_from(input_list):
    """
    remove stop words like 'of' or 'and'

    :param input_list: list of strings
    :return: list
    """
    stop_words = stopwords.words('english') + ['…', '&', '...']  # TODO: remove empty strings
    return [' '.join([e for e in w.split() if e not in stop_words]) for w in input_list]


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


# TODO: remove punctuations, numbers, links
def preprocess(input_str):
    """
    apply preprocessing steps on the input string

    :param input_str: string
    :return: string
    """
    input_list = str_to_list(input_str)
    normalized_list = normalize(input_list)
    cleaned_list = remove_stop_words_from(normalized_list)
    # check_spells_in(cleaned_list)  # view-only; no output yet
    return ', '.join(cleaned_list)
