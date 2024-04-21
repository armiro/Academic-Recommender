import os, pickle
import gensim.downloader as dl_api
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

CACHE_DIR = './cache/'


# create custom mapping over word combinations not available in pretrained model
def create_embedding_map_for(dataframe, col_name, model, map_dict):
    for ris in dataframe[col_name]:
        for ri in ris:
            if len(ri.split()) > 1:  # multiple-worded phrase only
                map_dict[ri] = model.get_mean_vector(ri.split())
    return map_dict


# generate word vector from model or embedding map, depending on being phrase or token
def generate_vectors_from(words, model, map_dict):
    vectors = list()
    for word in words:
        if len(word.split()) > 1:
            vectors.append(map_dict[word])
        else:
            vectors.append(model.get_vector(word))
    return vectors


# load model from the library (gensim or huggingface)
def load_model(library, model_name, model_file):
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
            with open("hf_token.txt", 'rt') as file:
                hf_token = file.read()
        except FileNotFoundError:
            print('huggingface auth token file <hf_token.txt> does not exist!')

        file_name = model_name[model_name.find('_')+1:] + '.txt'
        raw_model = hf_hub_download(repo_id=model_name, filename=file_name, cache_dir=CACHE_DIR,
                                    token=hf_token)
        model = KeyedVectors.load_word2vec_format(raw_model)

    else:
        raise ValueError("library name must be either 'gensim' or 'huggingface'")

    return model
