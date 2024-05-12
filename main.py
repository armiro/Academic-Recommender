"""
Academic Student-Student and Student-Professor Recommender using Word Embedding Models
Developed by Arman H. (https://github.com/armiro)
"""
import time
import logging
import pandas as pd

from preprocessing import preprocess
from utils.helper_functions import load_model

CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data.xlsx'

SPLIT_METHOD = 'token'  # must only be 'token'

MODEL_LIB = 'gensim'  # select between 'gensim' and 'huggingface'
MODEL_NAME = 'fasttext-wiki-news-subwords-300'  # gensim model
MODEL_FILE = MODELS_DIR + MODEL_NAME + '.pkl'  # if file available

STUDENT_ID = 6507
TOPN = 5
TARGET_ROLE = 'prof'  # select between 'student' and 'prof'

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
logging.getLogger('gensim').setLevel(logging.CRITICAL)


def load_data(data_path, method):
    """
    load dataset from xlsx/csv data and do preprocessing steps

    :param data_path: string, relational path to the dataset under data directory
    :param method: string, splitting method to tokenize strings; ['phrase' or 'token']
    :return: pandas dataframe, two dataframes: students df and professors df
    """
    xls_file = pd.ExcelFile(data_path)

    df_profs = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[1])
    df_students = pd.read_excel(xls_file, sheet_name=xls_file.sheet_names[0])

    # drop rows with empty name or research interests
    df_profs = df_profs.dropna(subset=['Name', 'Research Interests'], how='any')
    df_students = df_students.dropna(subset=['Name', 'Research Interests'], how='any')

    # apply preprocessing steps, defined separately
    df_profs['Tokenized RIs'] = df_profs['Research Interests'].apply(
        lambda x: preprocess(x, method=method))
    df_students['Tokenized RIs'] = df_students['Research Interests'].apply(
        lambda x: preprocess(x, method=method))

    return df_students, df_profs


def find_closest_to(student_idx, students, professors, model, topn, target_role):
    """
    find top matching students/professors for the input student using semantic similarity
    between research interests and previously trained word embedding model

    :param student_idx: int
    :param students: pandas dataframe
    :param professors: pandas dataframe
    :param model: word embedding model (previously trained)
    :param topn: int, number of suggestions
    :param target_role: string, target role to be suggested; ['student' or 'prof']
    :return: list of dicts
    """
    student_ris = students['Tokenized RIs'][student_idx]

    target_df = professors if target_role == 'prof' else students if target_role == 'student' else None
    if target_df is None:
        logging.error('invalid target role value: %s', target_role)
        raise ValueError("target role must either be 'prof' or 'student'")
    if target_role == 'student':  # drop input student if recommending students
        target_df = target_df.drop(student_idx)

    dists = []
    # iterate over each professor and their interests
    for target_id, target_ris in target_df['Tokenized RIs'].items():
        dist = model.wmdistance(document1=student_ris, document2=target_ris)
        dists.append((target_id, dist))

    sorted_dists = sorted(dists, key=lambda x: x[1], reverse=False)  # sort similarities
    # extract top n students/profs based on maximum similarity
    topn_targets = [target_df.loc[target_id] for target_id, _ in sorted_dists[:topn]]
    return topn_targets


def main():
    """
    load data and trained model, calculate research interest vector mapping, find top n
    most similar students/professors to a specific student

    :return: None
    """
    df_students, df_profs = load_data(data_path=DATA_DIR + DATASET_NAME, method=SPLIT_METHOD)

    logging.info('----------------------------')
    logging.info('loading word embedding model...')
    st = time.time()
    pretrained_model = load_model(library=MODEL_LIB, model_name=MODEL_NAME, model_file=MODEL_FILE)
    logging.info('model importing done!')
    logging.critical('elapsed time: %.2f secs', time.time() - st)

    logging.info('----------------------------')
    logging.info('Student Name: %s', df_students['Name'][STUDENT_ID])
    logging.info('Student Research Interests: %s', df_students['Research Interests'][STUDENT_ID])
    logging.info('Student Department: %s', df_students['University Field'][STUDENT_ID])

    logging.info('----------------------------')
    logging.info('searching for %ss ...', TARGET_ROLE)
    st = time.time()
    most_similar_targets = find_closest_to(student_idx=STUDENT_ID, students=df_students,
                                           professors=df_profs, model=pretrained_model,
                                           topn=TOPN, target_role=TARGET_ROLE)
    logging.info('search done!')
    logging.critical('elapsed time: %.2f secs', time.time() - st)
    logging.info('----------------------------')

    for idx, target in enumerate(most_similar_targets):
        logging.info('************')
        logging.info("best %s #%d: %s", TARGET_ROLE, idx+1, target['Name'])
        logging.info("%s research interests: %s", TARGET_ROLE, target['Research Interests'])
        logging.info("%s department: %s", TARGET_ROLE, target['University Field'])


if __name__ == "__main__":
    main()
