"""
Several useful functions
"""
import os
import logging

from sentence_transformers import SentenceTransformer

CACHE_DIR = './cache/'


def create_encoding_map_for(dataframe, col_name, model):
    """
    use sentence transformer to encode each set of research interests into a single
    encoded tensor

    :param dataframe: pandas dataframe
    :param col_name: string, dataframe column to perform encoding on
    :param model: sentence transformer model (pretrained)
    :return: list of tensors
    """
    encoding_map = {}
    for idx, ris in enumerate(dataframe[col_name]):
        encoding_map[idx] = model.encode(ris, batch_size=32, convert_to_tensor=True,
                                         show_progress_bar=False, precision='float32')
    return encoding_map


def load_model(model_name, model_file):
    """
    load pretrained model from the library (sentence transformers)

    :param model_name: string, model file name available in the library
    :param model_file: string, locally stored model file path
    :return: pretrained sentence transformer model
    """
    if os.path.exists(model_file):
        model = SentenceTransformer(model_name_or_path=model_file, cache_folder=CACHE_DIR)
    else:
        model = SentenceTransformer(model_name_or_path=model_name, cache_folder=CACHE_DIR)
        model.save(path=model_file)
    return model
