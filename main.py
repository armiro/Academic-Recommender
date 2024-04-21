import pandas as pd
import time

from preprocessing import preprocess
from utils.helper_functions import *


CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data.xlsx'

SPLIT_METHOD = 'phrase'  # select between 'phrase' and 'tokenize'

MODEL_LIB = 'gensim'  # select between 'gensim' and 'huggingface'
MODEL_NAME = 'fasttext-wiki-news-subwords-300'  # gensim model
# MODEL_NAME = 'Word2vec/wikipedia2vec_enwiki_20180420_100d'  # hf model
MODEL_FILE = MODELS_DIR + MODEL_NAME + '.pkl'  # if available

STUDENT_ID = 1700
TOPN = 5


def load_data(data_path, method):
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


def generate_ri_map(method, students, professors, model):
    if method == 'phrase':
        print('----------------------------')
        print('creating embedding map for phrases...')
        ri_map = dict()
        ri_map = create_embedding_map_for(students, col_name='Tokenized RIs',
                                          model=model, map_dict=ri_map)
        ri_map = create_embedding_map_for(professors, col_name='Tokenized RIs',
                                          model=model, map_dict=ri_map)
        print(f'there are {len(ri_map)} unique combination of words in the dataset.')
        return ri_map
    else:
        return None


def find_closest_profs_to(student_idx, students, professors, model, map_dict=None, topn=1):
    student_ris = students['Tokenized RIs'][student_idx]
    student_vecs = generate_vectors_from(student_ris, model=model, map_dict=map_dict)

    avg_sims = list()
    # iterate over each professor and their interests
    for prof_id, prof_ris in professors['Tokenized RIs'].items():
        sims = 0
        prof_vecs = generate_vectors_from(prof_ris, model=model, map_dict=map_dict)
        # calculate mean vector similarity between each student's interest and prof's interests
        for student_vec in student_vecs:
            sims += model.cosine_similarities(student_vec, prof_vecs).mean()

        avg_sim = sims/len(student_vecs)  # avg distance between student & prof
        avg_sims.append((prof_id, avg_sim))

    sorted_sims = sorted(avg_sims, key=lambda x: x[1], reverse=True)  # sort similarities
    # extract top n profs based on maximum similarity
    topn_profs = [professors.loc[prof_id] for prof_id, _ in sorted_sims[:topn]]
    return topn_profs


def main():
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

    # sanity check: all tokens (words) should be already present in the model vocab
    for interests in df_students['Tokenized RIs']:
        for interest in interests:
            for word in interest.split():
                if word not in pretrained_model: print(word)
    for interests in df_profs['Tokenized RIs']:
        for interest in interests:
            for word in interest.split():
                if word not in pretrained_model: print(word)

    ri_map = generate_ri_map(method=SPLIT_METHOD, students=df_students, professors=df_students,
                             model=pretrained_model)

    print('----------------------------')
    print('Student Name:', df_students['Name'][STUDENT_ID])
    print('Student Research Interests:', df_students['Research Interests'][STUDENT_ID])
    print('Student Department:', df_students['University Field'][STUDENT_ID])

    print('----------------------------')
    print('searching for professors ...')
    st = time.time()
    most_similar_profs = find_closest_profs_to(student_idx=STUDENT_ID, students=df_students,
                                               professors=df_profs, model=pretrained_model,
                                               map_dict=ri_map, topn=TOPN)
    print('search done!')
    print('elapsed time:', round(time.time() - st), 'secs')
    print('----------------------------')

    for prof_idx, top_prof in enumerate(most_similar_profs):
        print('******++')
        print(f"Best Professor #{prof_idx + 1}: {top_prof['Name']}")
        print(f"Prof Research Focus: {top_prof['Research Interests']}")
        print(f"Prof Faculty: {top_prof['University Field']}")


if __name__ == "__main__":
    main()
