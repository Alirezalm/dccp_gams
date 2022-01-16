import sys

from problem_class import SparseLogReg


def run_experiment(problem_data):
    dslr = SparseLogReg(problem_data = problem_data)
    dslr.generate_data(nSamples = int(problem_data['nSamples']))
    dslr.create_variables()
    dslr.create_objective()
    dslr.create_constraints(5)
    res = dslr.solve(problem_data['solver'])
    print(res)


if __name__ == '__main__':
    args = [int(arg) for arg in sys.argv[2:]]
    data = {
        'name': "dslr",
        'nSamples': args[0],
        'nVars': args[1],
        'nZeros': args[2],
        'nNodes': args[3],
        'solver': sys.argv[1]
    }
    run_experiment(problem_data = data)
