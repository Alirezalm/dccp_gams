import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from problem_class import SparseLogReg

app = Flask(__name__)
PATH = os.path.join(os.path.dirname(__file__))
ENV_PATH = os.path.join(PATH, '.env')

load_dotenv(ENV_PATH)


@app.route('/', methods = ['POST', 'GET'])
def main_page():
    if request.method == 'POST':
        problem_data = json.loads(request.get_data())
        dslr = SparseLogReg(problem_data)
        dslr.generate_data(nSamples = int(problem_data['nSamples']))
        dslr.create_variables()
        dslr.create_objective()
        dslr.create_constraints(5)
        dslr.solve()
        return jsonify({'status': 55})

    else:
        return jsonify({'status': 1})


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = '5050')
