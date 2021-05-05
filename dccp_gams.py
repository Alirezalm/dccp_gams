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

    @abstractmethod
    def solve(self, solver):
        pass
