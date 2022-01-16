import abc
from abc import ABC, abstractmethod

from pyomo.environ import *

PROBLEM_CLASS = {
    'distributedSparseLogisticRegression': 'dslr',
    'distributedSparseQCQP': 'dsqcqp'
}


class RandDCCP(ABC):

    def __init__(self, problem_data):
        self.name = problem_data['name']
        self.nVars = int(problem_data['nVars'])
        self.nZeros = int(problem_data['nZeros'])
        self.nNodes = int(problem_data['nNodes'])
        self.model = None

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def create_variables(self):
        pass

    @abstractmethod
    def create_objective(self):
        pass

    @abstractmethod
    def create_constraints(self, bound):
        pass

    def solve(self, solver_name):
        # past = solver_name
        # if solver_name in ['gurobi', 'cplex', 'shot']:
        #     solver_name = None
        solver = SolverFactory('gams')
        shot_opts = ["$onecho > shot.opt",  "Dual.MIP.Solver=2", "Dual.TreeStrategy=1", "Model.Convexity.AssumeConvex=1", "$offecho"]
        options = ['GAMS_MODEL.reslim = 600;', 'GAMS_MODEL.optcr = 0.05;', 'GAMS_MODEL.optfile = 1;'] + shot_opts
        print(f'SOLVING USING {solver_name}\nProblem size: {self.nVars}')
        results = solver.solve(self.model, solver = solver_name, tee = True, keepfiles = True, add_options = options)
        print(results)
        lower_bound = results.problem.lower_bound
        upper_bound = results.problem.upper_bound
        elapsed_time = results.solver.user_time
        status = results.solver.termination_condition
        gap = (upper_bound - lower_bound) / abs(upper_bound + 1e-8)

        out = {
            'solver': solver_name,
            'gap': gap,
            'time': elapsed_time,
            'status': status
        }
        return out
