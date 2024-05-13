"""
Academic Student-Student and Student-Professor Recommender using Word Embedding Models
Developed by Arman H. (https://github.com/armiro)
"""
import time
import logging
import pandas as pd

from preprocessing import preprocess
from utils.helper_functions import create_encoding_map_for, load_model
from sentence_transformers import util
from transformers.utils import logging as tlogging

CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data.xlsx'

MODEL_NAME = 'all-MiniLM-L12-v2'
MODEL_FILE = MODELS_DIR + MODEL_NAME  # if folder available

STUDENT_ID = 6507
TOPN = 5
TARGET_ROLE = 'prof'  # select between 'student' and 'prof'

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
tlogging.disable_progress_bar()  # disable tqdm bar when encoding inputs


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
    df_profs['Preprocessed RIs'] = df_profs['Research Interests'].apply(lambda x: preprocess(x))
    df_students['Preprocessed RIs'] = df_students['Research Interests'].apply(lambda x: preprocess(x))

    return df_students, df_profs


def find_closest_to(student_idx, students, professors, model, map_dict, topn, target_role):
    """
    find top matching students/professors for the input student using semantic similarity
    between research interests and previously trained word embedding model

    :param student_idx: int
    :param students: pandas dataframe
    :param professors: pandas dataframe
    :param model: word embedding model - sentence transformer (pretrained)
    :param map_dict: dict, encoding map of all data records
    :param topn: int, number of suggestions
    :param target_role: string, target role to be suggested; ['student' or 'prof']
    :return: list of dicts
    """
    student_ris = students['Preprocessed RIs'][student_idx]
    student_embd = model.encode(student_ris, convert_to_tensor=True)

    target_df = professors if target_role == 'prof' else students if target_role == 'student' else None
    if target_df is None:
        logging.error('invalid target role value: %s', target_role)
        raise ValueError("target role must either be 'prof' or 'student'")
    if target_role == 'student':  # drop input student if recommending students
        target_df = target_df.drop(student_idx)

    cos_sims = []
    # iterate over each target (student/professor)
    for target_id, target_embd in map_dict.items():
        # calculate cosine similarity between student and target embeddings
        cos_sim = util.pytorch_cos_sim(student_embd, target_embd)
        cos_sims.append((target_id, cos_sim))

    sorted_sims = sorted(cos_sims, key=lambda x: x[1], reverse=True)  # sort similarities
    # extract top n students/profs based on maximum similarity
    topn_targets = [target_df.loc[target_id] for target_id, _ in sorted_sims[:topn]]
    return topn_targets


def main():
    """
    load data and trained model, calculate research interest vector mapping, find top n
    most similar students/professors to a specific student

    :return: None
    """
    df_students, df_profs = load_data(data_path=DATA_DIR + DATASET_NAME)

    logging.info('----------------------------')
    logging.info('loading word embedding model...')
    st = time.time()
    pretrained_model = load_model(model_name=MODEL_NAME, model_file=MODEL_FILE)
    logging.info('model importing done!')
    logging.critical('elapsed time: %.2f secs', time.time() - st)

    st = time.time()
    logging.info('creating encoding map using sentence transformer')
    encoding_map = create_encoding_map_for(dataframe=df_profs, col_name='Preprocessed RIs',
                                           model=pretrained_model)
    logging.info('encoding map created!')
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
                                           map_dict=encoding_map, topn=TOPN, target_role=TARGET_ROLE)
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
