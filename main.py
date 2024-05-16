"""
Academic Student-Student and Student-Professor Recommender using Word Embedding Models
Developed by Arman H. (https://github.com/armiro)
"""
import os
import time
import logging
import pandas as pd
import torch

from preprocessing import preprocess
from utils.helper_functions import create_encoding_map_for, load_model


CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data.xlsx'

MODEL_NAME = 'bert-base-uncased'  # sentence transformer model
MODEL_DIR = MODELS_DIR + MODEL_NAME  # if saved folder available

STUDENT_ID = 6507
TOPN = 5
TARGET_ROLE = 'prof'  # select between 'student' and 'prof'

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)


def load_data(data_path):
    """
    load dataset from xlsx/csv data and do preprocessing steps

    :param data_path: string, relational path to the dataset under data directory
    :return: pandas dataframe, two dataframes: students df and professors df
    """
    xls_file = pd.ExcelFile(data_path)

    df_profs = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[1])
    df_students = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[0])

    # drop rows with empty name or research interests
    df_profs = df_profs.dropna(subset=['Name', 'Research Interests'], how='any')
    df_students = df_students.dropna(subset=['Name', 'Research Interests'], how='any')

    # apply preprocessing steps, defined separately
    df_profs['Preprocessed RIs'] = df_profs['Research Interests'].apply(preprocess)
    df_students['Preprocessed RIs'] = df_students['Research Interests'].apply(preprocess)

    return df_students, df_profs


def find_closest_to(this_student, dataframe, model, tokenizer, map_dict, topn):
    """
    find top matching students/professors for the input student using semantic similarity
    between research interests and previously trained word embedding model

    :param this_student: string
    :param dataframe: pandas dataframe
    :param model: word embedding model - BERT (pretrained)
    :param tokenizer: tokenizer - BERT (pretrained)
    :param map_dict: dict, encoding map of target dataframe records
    :param topn: int, number of suggestions
    :return: list of dicts
    """
    student_tokens = tokenizer.encode(this_student, return_tensors='pt')
    student_embd = model(student_tokens)[0].mean(dim=1)

    cos_sims = []
    # iterate over each target (student/professor) embedding
    for target_id, target_embd in map_dict.items():
        # calculate cosine similarity between student and target embeddings
        cos_sim = torch.cosine_similarity(student_embd, target_embd)
        cos_sims.append((target_id, cos_sim))

    sorted_sims = sorted(cos_sims, key=lambda x: x[1], reverse=True)  # sort similarities
    topn_targets = [dataframe.loc[idx] for idx, _ in sorted_sims[:topn]]  # extract top n
    return topn_targets


def main():
    """
    load data and trained model, calculate research interest vector mapping, find top n
    most similar students/professors to a specific student

    :return: None
    """
    df_students, df_profs = load_data(data_path=DATA_DIR + DATASET_NAME)
    student_ris = df_students['Preprocessed RIs'][STUDENT_ID]
    target_df = df_profs if TARGET_ROLE == 'prof' else df_students if TARGET_ROLE == 'student' else None

    if target_df is None:
        logging.error('invalid target role value: %s', TARGET_ROLE)
        raise ValueError("target role must either be 'prof' or 'student'")

    logging.info('----------------------------')
    logging.info('loading BERT model and tokenizer ...')
    st = time.time()
    pretrained_model, tokenizer = load_model(model_name=MODEL_NAME, model_dir=MODEL_DIR)
    logging.info('model importing done!')
    logging.critical('elapsed time: %.2f secs', time.time() - st)

    st = time.time()
    logging.info('creating encoding map for target dataset ...')
    cache_file = f"{os.path.splitext(DATASET_NAME)[0]},{MODEL_NAME},{TARGET_ROLE}.pkl"
    encoding_map = create_encoding_map_for(dataframe=target_df, col_name='Preprocessed RIs',
                                           model=pretrained_model, tokenizer=tokenizer,
                                           cache_file=cache_file)
    logging.info('encoding map created!')
    logging.critical('elapsed time: %.2f secs', time.time() - st)

    logging.info('----------------------------')
    logging.info('Student Name: %s', df_students['Name'][STUDENT_ID])
    logging.info('Student Research Interests: %s', df_students['Research Interests'][STUDENT_ID])
    logging.info('Student Department: %s', df_students['University Field'][STUDENT_ID])

    if TARGET_ROLE == 'student':  # drop ref student from df/dict if recommending students
        target_df = target_df.drop(STUDENT_ID)
        encoding_map.pop(STUDENT_ID, None)

    logging.info('----------------------------')
    logging.info('searching for %ss ...', TARGET_ROLE)
    st = time.time()
    most_similar_targets = find_closest_to(this_student=student_ris, dataframe=target_df, topn=TOPN,
                                           model=pretrained_model, tokenizer=tokenizer,
                                           map_dict=encoding_map)
    logging.info('search done!')
    logging.critical('elapsed time: %.2f secs', time.time() - st)
    logging.info('----------------------------')

    for idx, target in enumerate(most_similar_targets):
        logging.info('************')
        logging.info("best %s #%d: %s", TARGET_ROLE, idx + 1, target['Name'])
        logging.info("%s research interests: %s", TARGET_ROLE, target['Research Interests'])
        logging.info("%s department: %s", TARGET_ROLE, target['University Field'])


if __name__ == "__main__":
    main()
