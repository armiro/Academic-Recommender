"""
Several useful functions
"""
import os
import pickle
import logging
import gensim.downloader as dl_api
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

CACHE_DIR = './cache/'


def load_model(library, model_name, model_file):
    """
    load pretrained model from the library (gensim or huggingface)

    :param library: string, ['gensim' or 'huggingface']
    :param model_name: string, model file name available in the library
    :param model_file: string, locally stored model file address
    :return: pretrained word embedding model
    """
    if library == 'gensim':
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        else:
            model = dl_api.load(name=model_name)
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

    elif library == 'huggingface':

        try:
            with open("hf_token.txt", mode='rt', encoding='utf-8') as file:
                hf_token = file.read()
        except FileNotFoundError:
            logging.exception('huggingface auth token file <hf_token.txt> not found!')

        file_name = model_name[model_name.find('_')+1:] + '.txt'
        raw_model = hf_hub_download(repo_id=model_name, filename=file_name, cache_dir=CACHE_DIR,
                                    token=hf_token)
        model = KeyedVectors.load_word2vec_format(raw_model)

    else:
        logging.error('invalid library name: %s', library)
        raise ValueError("library name must be either 'gensim' or 'huggingface'")

    return model
