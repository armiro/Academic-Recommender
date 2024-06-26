"""
Local Flask API
"""
import argparse
import logging
from flask import Flask, request, jsonify
from main import find_closest_to, load_data, generate_ri_map
from utils.helper_functions import load_model


app = Flask(__name__)

CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data.xlsx'

DEFAULT_SPLIT_METHOD = 'phrase'

DEFAULT_MODEL_LIB = 'gensim'
DEFAULT_MODEL_NAME = 'fasttext-wiki-news-subwords-300'


@app.before_request
def explain():
    """
    print initializing text before processing API request

    :return: None
    """
    logging.info('receiving top matching professors for target student...')


@app.route('/api/top_professors', methods=['GET', 'POST'])
def get_topn_professors():
    """
    main API run. parse url arguments and respond with top n suggestions for input student

    :return: dict (json)
    """
    try:
        student_id = int(request.args.get('student_id'))
        topn = int(request.args.get('topn', 1))  # num suggestions; default to 1
        target = request.args.get('recommend')
        split_method = request.args.get('method', DEFAULT_SPLIT_METHOD)  # word splitting method
        window_size = int(request.args.get('winsize', 5))  # window size for cluster embedding
        model_file = MODELS_DIR + args.model_name + '.pkl'  # preloaded model file (if available)

        df_students, df_profs = load_data(data_path=DATA_DIR + args.dataset, method=split_method)
        pretrained_model = load_model(library=args.model_lib, model_name=args.model_name,
                                      model_file=model_file)
        ri_map = generate_ri_map(method=split_method, students=df_students, professors=df_students,
                                 model=pretrained_model, window_size=window_size)
        top_profs = find_closest_to(student_idx=student_id, students=df_students,
                                    professors=df_profs, model=pretrained_model, map_dict=ri_map,
                                    topn=topn, target_role=target)

        # format the response as json
        serialized_top_profs = [prof.drop('Tokenized RIs').to_dict() for prof in top_profs]
        response = {'student': df_students.iloc[student_id].drop('Tokenized RIs').to_dict(),
                    'top_profs': serialized_top_profs}
        return jsonify(response), 200

    except Exception as error_msg:
        return jsonify({'error': str(error_msg)}), 400


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process cli arguments')
    parser.add_argument('-dataset', type=str, default=DATASET_NAME, help='dataset file name')
    parser.add_argument('-model_lib', type=str, default=DEFAULT_MODEL_LIB,
                        help='library for the word embedding model: "gensim" or "huggingface"')
    parser.add_argument('-model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='name of the pretrained word embedding model')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=105, debug=True)
