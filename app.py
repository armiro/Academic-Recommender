"""
Local Flask API
"""
import os
import argparse
import logging
from flask import Flask, request, jsonify
from main import find_closest_to, load_data
from utils.helper_functions import load_model, create_encoding_map_for


app = Flask(__name__)

CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data_gs.xlsx'

DEFAULT_MODEL_NAME = 'all-MiniLM-L12-v2'


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
        model_dir = MODELS_DIR + args.model_name  # preloaded model folder (if available)

        df_students, df_profs = load_data(data_path=DATA_DIR + args.dataset)
        student_ris = df_students['Preprocessed RIs'][student_id]
        target_df = df_profs if target == 'prof' else df_students if target == 'student' else None

        pretrained_model = load_model(model_name=args.model_name, model_file=model_dir)
        cache_file = f"{os.path.splitext(DATASET_NAME)[0]},{args.model_name},{target}.pkl"
        encoding_map = create_encoding_map_for(dataframe=target_df, col_name='Preprocessed RIs',
                                               model=pretrained_model, cache_file=cache_file)
        top_profs = find_closest_to(this_student=student_ris, dataframe=target_df, topn=topn,
                                    model=pretrained_model, map_dict=encoding_map)

        # format the response as json
        serialized_top_profs = [prof.drop('Preprocessed RIs').to_dict() for prof in top_profs]
        response = {'student': df_students.iloc[student_id].drop('Preprocessed RIs').to_dict(),
                    'top_profs': serialized_top_profs}
        return jsonify(response), 200

    except Exception as error_msg:
        return jsonify({'error': str(error_msg)}), 400


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process cli arguments')
    parser.add_argument('-dataset', type=str, default=DATASET_NAME, help='dataset file name')
    parser.add_argument('-model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='name of the pretrained word embedding model')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=105, debug=True)
