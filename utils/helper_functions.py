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


def create_embedding_map_for(dataframe, col_name, model, map_dict):
    """
    create custom mapping over word combinations not available in pretrained model
    by averaging its sub-word vectors

    :param dataframe: pandas dataframe
    :param col_name: string, dataframe column to perform mapping on
    :param model: word embedding model (previously trained)
    :param map_dict: dict
    :return: dict
    """
    for ris in dataframe[col_name]:
        for ri in ris:
            if len(ri.split()) > 1:  # multiple-worded phrase only
                map_dict[ri] = model.get_mean_vector(ri.split())
    return map_dict


def create_cluster_map_for(dataframe, target_col, ref_col, model, map_dict, window_size=5):
    """
    create mapping for phrases not available in pretrained model by iterating over
    clusters created from unique values in target_col, and averaging neighboring phrases

    :param dataframe: pandas dataframe
    :param target_col: string, dataframe column to perform mapping on
    :param ref_col: string, dataframe column to use as clusters
    :param model: word embedding model (previously trained)
    :param window_size: integer, num neighboring phrases to look at
    :param map_dict: dict
    :return: dict
    """
    uniq_ref_values = dataframe[ref_col].unique()

    # perform mapping for each cluster, based on neighboring phrases only in that cluster
    for ref_val in uniq_ref_values:
        ref_df = dataframe.loc[dataframe[ref_col] == ref_val, target_col]
        ref_phrases = list(set(sum(ref_df.tolist(), [])))

        for idx, phrase in enumerate(ref_phrases):
            if phrase not in model.key_to_index:  # multi-word or unseen phrases
                start_idx = max(0, idx - window_size)
                end_idx = min(len(ref_phrases), idx + window_size + 1)
                context_phrases = ref_phrases[start_idx:end_idx]
                word_vecs = []

                for context_phrase in context_phrases:
                    for word in context_phrase.split():
                        if word in model:
                            word_vecs.append(model[word])

                map_dict[phrase] = sum(word_vecs)/len(word_vecs)

    return map_dict


def generate_vectors_from(words, model, map_dict):
    """
    generate word vector from pretrained model or constructed embedding map, depending on
    word being available in model's vocab

    :param words: list of strings
    :param model: word embedding model (previously trained)
    :param map_dict: dict, previously generated word embedding map for multi-word phrases
    :return: list
    """
    vectors = []
    for word in words:
        if word not in model.key_to_index:
            vectors.append(map_dict[word])
        else:
            vectors.append(model.get_vector(word))
    return vectors


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
