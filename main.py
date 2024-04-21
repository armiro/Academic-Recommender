import pandas as pd
# import numpy as np

import time


from preprocessing import preprocess
from utils.helper_functions import *


CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
SPLIT_METHOD = 'phrase'  # select between 'phrase' and 'tokenize'
MODEL_LIB = 'gensim'  # select between 'gensim' and 'huggingface'

xls_file = pd.ExcelFile(DATA_DIR + 'university_data.xlsx')
print(xls_file.sheet_names)

df_profs = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[1])
df_students = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[0])

df_profs.dropna(axis=0, inplace=True)  # drop empty rows
df_students.dropna(axis=0, inplace=True)  # drop empty rows

# sanity check over null values
assert df_profs.all().all()
assert df_students.all().all()


df_profs['Tokenized RIs'] = df_profs['Research Interests'].apply(
    lambda x: preprocess(x, method=SPLIT_METHOD))
df_students['Tokenized RIs'] = df_students['Research Interests'].apply(
    lambda x: preprocess(x, method=SPLIT_METHOD))

print(df_profs.head())
print(df_students.head())


available_corpora = dl_api.info()['models']

# see all models and their file sizes
for name, metadata in available_corpora.items():
    if not name.startswith('_'):
        print(f"name: {name}, size: {round(metadata['file_size']/(1024.0*1024.0))} MB")


print('loading word embedding model...')
st = time.time()
model_name = 'fasttext-wiki-news-subwords-300'
model_file = MODELS_DIR + model_name + '.pkl'
# model_name = 'Word2vec/wikipedia2vec_enwiki_20180420_100d'
pretrained_model = load_model(library=MODEL_LIB, model_name=model_name, model_file=model_file)
print('importing the model done!')
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


if SPLIT_METHOD == 'phrase':
    print('creating embedding map for phrases...')
    ri_map = dict()
    ri_map = create_embedding_map_for(df_students, col_name='Tokenized RIs', model=pretrained_model,
                                      map_dict=ri_map)
    ri_map = create_embedding_map_for(df_profs, col_name='Tokenized RIs', model=pretrained_model,
                                      map_dict=ri_map)
    print(f'there are {len(ri_map)} unique combination of words in the dataset.')
else:
    ri_map = None


def find_closest_profs_to(student_idx, students, professors, model, map_dict=None, topn=1):
    student_ris = students['Tokenized RIs'][student_idx]
    student_vecs = generate_vectors_from(student_ris, model=model, map_dict=map_dict)

    avg_sims = list()
    # iterate over each professor and their interests
    for prof_id, prof_ris in professors['Tokenized RIs'].items():
        sims = 0
        prof_vecs = generate_vectors_from(prof_ris, model=model, map_dict=map_dict)
        # calculate vector distance between each student's interest and prof's interests
        for student_vec in student_vecs:
            sims += pretrained_model.cosine_similarities(student_vec, prof_vecs).mean()

        avg_sim = sims/len(student_vecs)  # avg distance between student & prof
        avg_sims.append((prof_id, avg_sim))

    sorted_sims = sorted(avg_sims, key=lambda x: x[1], reverse=True)  # sort distances (descending)
    # extract top n profs based on minimum distance
    topn_profs = [professors.loc[prof_id] for prof_id, _ in sorted_sims[:topn]]
    return topn_profs


idx = 1706
print('Student Name:', df_students['Name'][idx])
print('Student Research Interests:', df_students['Research Interests'][idx])
print('Student Department:', df_students['University Field'][idx])

print('----------------------------')
print('searching for professors ...')
st = time.time()
most_similar_profs = find_closest_profs_to(student_idx=idx, students=df_students, professors=df_profs,
                                           model=pretrained_model, map_dict=ri_map, topn=5)
print('search done!')
print('elapsed time:', round(time.time() - st), 'secs')
print('----------------------------')

for prof_idx, top_prof in enumerate(most_similar_profs):
    print('******')
    print(f"Best Professor #{prof_idx+1}: {top_prof['Name']}")
    print(f"Prof Research Focus: {top_prof['Research Interests']}")
    print(f"Prof Faculty: {top_prof['University Field']}")
