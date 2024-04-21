from flask import Flask, request, jsonify
from main import find_closest_profs_to, load_data, generate_ri_map
from utils.helper_functions import load_model

app = Flask(__name__)

CACHE_DIR = './cache/'
DATA_DIR = './data/'
MODELS_DIR = './models/'
DATASET_NAME = 'university_data.xlsx'

SPLIT_METHOD = 'phrase'

MODEL_LIB = 'gensim'
MODEL_NAME = 'fasttext-wiki-news-subwords-300'
MODEL_FILE = MODELS_DIR + MODEL_NAME + '.pkl'


@app.route('/api/top_professors', methods=['GET', 'POST'])
def get_topn_professors():
    try:
        student_id = int(request.args.get('student_id'))
        topn = int(request.args.get('topn', 1))  # default to 1 if topn not provided

        df_students, df_profs = load_data(data_path=DATA_DIR + DATASET_NAME, method=SPLIT_METHOD)
        pretrained_model = load_model(library=MODEL_LIB, model_name=MODEL_NAME, model_file=MODEL_FILE)
        ri_map = generate_ri_map(method=SPLIT_METHOD, students=df_students, professors=df_students,
                                 model=pretrained_model)
        top_profs = find_closest_profs_to(student_idx=student_id, students=df_students, professors=df_profs,
                                          model=pretrained_model, map_dict=ri_map, topn=topn)

        # format the response as json
        serialized_top_profs = [prof.to_dict() for prof in top_profs]
        response = dict()
        response['student'] = df_students.iloc[student_id].to_dict()
        response['top_profs'] = serialized_top_profs
        return jsonify(response), 200

    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105, debug=True)

