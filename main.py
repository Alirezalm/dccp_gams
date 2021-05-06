import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from problem_class import SparseLogReg, SparseQCQP

app = Flask(__name__)
PATH = os.path.join(os.path.dirname(__file__))
ENV_PATH = os.path.join(PATH, '.env')

load_dotenv(ENV_PATH)


@app.route('/', methods = ['POST', 'GET'])
def main_page():
    if request.method == 'POST':
        problem_data = json.loads(request.get_data())
        if problem_data['name'] == 'dslr':
            dslr = SparseLogReg(problem_data)
            dslr.generate_data(nSamples = int(problem_data['nSamples']))
            dslr.create_variables()
            dslr.create_objective()
            dslr.create_constraints(5)
            solver = problem_data['selected_solver']
            results = dslr.solve(solver)
            return jsonify(results)
        else:
            dsqcqp = SparseQCQP(problem_data = problem_data)
            dsqcqp.generate_data()
            dsqcqp.create_objective()
            dsqcqp.create_constraints(5)
            solver = problem_data['selected_solver']
            results = dsqcqp.solve(solver)
            return jsonify(results)
    else:
        return jsonify({'status': 1})


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = '5050')
