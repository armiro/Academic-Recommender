"""
Several useful functions
"""
import os
import pickle

from transformers import BertModel, BertTokenizer
import torch

CACHE_DIR = './cache/'


def create_encoding_map_for(dataframe, col_name, model, tokenizer, cache_file):
    """
    use sentence transformer to encode each set of research interests into a single
    encoded tensor

    :param dataframe: pandas dataframe
    :param col_name: string, dataframe column to perform encoding on
    :param model: BERT model (pretrained)
    :param tokenizer: BERT tokenizer (pretrained)
    :param cache_file: path to cached embeddings file (if available)
    :return: list of tensors
    """
    cache_path = CACHE_DIR + cache_file
    map_dict = {}
    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as file:
            map_dict = pickle.load(file)
    else:
        with torch.no_grad():
            for idx, ris in enumerate(dataframe[col_name]):
                tokens = tokenizer.encode(ris, return_tensors='pt')
                map_dict[idx] = model(tokens)[0].mean(dim=1)
        # store embeddings in cache directory
        with open(cache_path, mode='wb') as file:
            pickle.dump(map_dict, file)
    return map_dict


def load_model(model_name, model_dir):
    """
    load pretrained BERT model from the library

    :param model_name: string, model file name available in the library
    :param model_dir: string, locally stored model/tokenizer folder path
    :return: pretrained sentence transformer model
    """
    if os.path.exists(model_dir):
        tokenizer = BertTokenizer.from_pretrained(model_dir + '/tokenizer/')
        model = BertModel.from_pretrained(model_dir + '/model/')
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir+'/tokenizer/')
        model.save_pretrained(model_dir + '/model/')

    return model, tokenizer
