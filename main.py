"""
Academic Student-Student and Student-Professor Recommender using Word Embedding Models
Developed by Arman H. (https://github.com/armiro)
"""
import time
import pandas as pd

import gensim.downloader as dl_api
from preprocessing import preprocess
from utils.helper_functions import *


CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data.xlsx'

SPLIT_METHOD = 'phrase'  # select between 'phrase' and 'token'
WINDOW_SIZE = 5  # window size for cluster mapping

MODEL_LIB = 'gensim'  # select between 'gensim' and 'huggingface'
MODEL_NAME = 'glove-wiki-gigaword-50'  # gensim model
# MODEL_NAME = 'Word2vec/wikipedia2vec_enwiki_20180420_100d'  # hf model
MODEL_FILE = MODELS_DIR + MODEL_NAME + '.pkl'  # if file available

STUDENT_ID = 886
TOPN = 5
TARGET_ROLE = 'student'  # select between 'student' and 'prof'


def load_data(data_path, method):
    """
    load dataset from xlsx/csv data and do preprocessing steps

    :param data_path: string, relational path to the dataset under data directory
    :param method: string, splitting method to tokenize strings; ['phrase' or 'token']
    :return: pandas dataframe, two dataframes: students df and professors df
    """
    xls_file = pd.ExcelFile(data_path)
    # print(xls_file.sheet_names)

    df_profs = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[1])
    df_students = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[0])

    df_profs.dropna(axis=0, inplace=True)  # drop empty rows
    df_students.dropna(axis=0, inplace=True)  # drop empty rows

    # sanity check over null values
    assert df_profs.all().all()
    assert df_students.all().all()

    # apply preprocessing steps, defined separately
    df_profs['Tokenized RIs'] = df_profs['Research Interests'].apply(
        lambda x: preprocess(x, method=method))
    df_students['Tokenized RIs'] = df_students['Research Interests'].apply(
        lambda x: preprocess(x, method=method))

    # print(df_profs.head())
    # print(df_students.head())
    return df_students, df_profs


def check_for_unseen_words_in(dataframe_col, model):
    have_unseen_words = False
    for interests in dataframe_col:
        for interest in interests:
            if have_unseen_words: break
            for word in interest.split():
                if word not in model:
                    have_unseen_words = True
                    break
    return have_unseen_words


def generate_ri_map(method, students, professors, model, window_size):
    """
    generate mean word embedding map for every phrase in the dataset, based on sub-word
    averaging or neighboring words averaging (depending on use_cluster_mapping)
    activated only if SPLIT_METHOD == 'phrase'

    :param method: string, splitting method
    :param students: pandas dataframe
    :param professors: pandas dataframe
    :param model: word embedding model (previously trained)
    :param window_size: integer, window size when averaging neighboring vectors
    :return: dict or none, depending on 'method' value
    """
    have_unseen_words = check_for_unseen_words_in(pd.concat([students['Tokenized RIs'],
                                                             professors['Tokenized RIs']]),
                                                  model=model)
    # if method == 'token' and have_unseen_words:
    #     raise ValueError('there are words not available in ')
    if method == 'phrase':
        print('----------------------------')
        print('creating embedding map for phrases...')
        ri_map = {}
        if have_unseen_words:
            print('using cluster mapping; there are words not available in model vocab...')
            ri_map = create_cluster_map_for(students, target_col='Tokenized RIs',
                                            ref_col='University Field', model=model,
                                            map_dict=ri_map, window_size=window_size)
            ri_map = create_cluster_map_for(professors, target_col='Tokenized RIs',
                                            ref_col='University Field', model=model,
                                            map_dict=ri_map, window_size=window_size)
        else:
            print('using sub-word mean mapping; all sub-words are available in model vocab...')
            ri_map = create_embedding_map_for(students, col_name='Tokenized RIs',
                                              model=model, map_dict=ri_map)
            ri_map = create_embedding_map_for(professors, col_name='Tokenized RIs',
                                              model=model, map_dict=ri_map)
        print(f'there are {len(ri_map)} unique unseen words/phrases in the dataset.')
        return ri_map

    return None


def find_closest_to(student_idx, students, professors, model, map_dict=None, topn=1, target_role='student'):
    """
    find top matching students/professors for the input student using semantic similarity
    between research interests and previously trained word embedding model

    :param student_idx: int
    :param students: pandas dataframe
    :param professors: pandas dataframe
    :param model: word embedding model (previously trained)
    :param map_dict: dict or none, map of research interest vectors if split method is 'phrase'
    :param topn: int, number of suggestions
    :param target_role: string, target role to be suggested; ['student' or 'prof']
    :return: list of dicts
    """
    student_ris = students['Tokenized RIs'][student_idx]
    student_vecs = generate_vectors_from(student_ris, model=model, map_dict=map_dict)

    target_df = professors if target_role == 'prof' else students if target_role == 'student' else None
    if target_df is None:
        raise ValueError("target role must either be 'prof' or 'student'")
    if target_role == 'student':  # drop input student if recommending students
        target_df = target_df.drop(student_idx)

    avg_sims = []
    # iterate over each professor and their interests
    for target_id, target_ris in target_df['Tokenized RIs'].items():
        sims = 0
        target_vecs = generate_vectors_from(target_ris, model=model, map_dict=map_dict)
        # calculate mean vector similarity between each student's interest and target's interests
        for student_vec in student_vecs:
            sims += model.cosine_similarities(student_vec, target_vecs).mean()

        avg_sim = sims/len(student_vecs)  # avg distance between student & student/prof
        avg_sims.append((target_id, avg_sim))

    sorted_sims = sorted(avg_sims, key=lambda x: x[1], reverse=True)  # sort similarities
    # extract top n students/profs based on maximum similarity
    topn_targets = [target_df.loc[target_id] for target_id, _ in sorted_sims[:topn]]
    return topn_targets


def main():
    """
    load data and trained model, calculate research interest vector mapping, find top n
    most similar students/professors to a specific student

    :return: None
    """
    df_students, df_profs = load_data(data_path=DATA_DIR + DATASET_NAME, method=SPLIT_METHOD)
    available_corpora = dl_api.info()['models']

    # see all models and their file sizes
    print('*** list of pretrained models in gensim library ***')
    for name, metadata in available_corpora.items():
        if not name.startswith('_'):
            print(f"name: {name}, size: {round(metadata['file_size']/(1024*1024))} MB")

    print('----------------------------')
    print('loading word embedding model...')
    st = time.time()
    pretrained_model = load_model(library=MODEL_LIB, model_name=MODEL_NAME, model_file=MODEL_FILE)
    print('model importing done!')
    print('elapsed time:', round(time.time() - st), 'secs')

    ri_map = generate_ri_map(method=SPLIT_METHOD, students=df_students, professors=df_students,
                             model=pretrained_model, window_size=WINDOW_SIZE)

    print('----------------------------')
    print('Student Name:', df_students['Name'][STUDENT_ID])
    print('Student Research Interests:', df_students['Research Interests'][STUDENT_ID])
    print('Student Department:', df_students['University Field'][STUDENT_ID])

    print('----------------------------')
    print(f'searching for {TARGET_ROLE}s ...')
    st = time.time()
    most_similar_targets = find_closest_to(student_idx=STUDENT_ID, students=df_students,
                                           professors=df_profs, model=pretrained_model,
                                           map_dict=ri_map, topn=TOPN, target_role=TARGET_ROLE)
    print('search done!')
    print('elapsed time:', round(time.time() - st), 'secs')
    print('----------------------------')

    for target_idx, top_target in enumerate(most_similar_targets):
        print('******++')
        print(f"Best {TARGET_ROLE.capitalize()} #{target_idx + 1}: {top_target['Name']}")
        print(f"{TARGET_ROLE.capitalize()} Research Interests: {top_target['Research Interests']}")
        print(f"{TARGET_ROLE.capitalize()} Department: {top_target['University Field']}")


if __name__ == "__main__":
    main()
