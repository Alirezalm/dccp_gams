from numpy import eye, zeros
from numpy.random import rand
from pyomo.opt.results import results_
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell
from scipy import randn
from sklearn import preprocessing
from pyomo.environ import *

from dccp_gams import RandDCCP
import pyutilib.subprocess.GlobalData

pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False


class SparseLogReg(RandDCCP):
    dataset = None
    response = None

    def generate_data(self, nSamples = 100):
        if nSamples == 100:
            print("DEFAULT NUMBER OF SAMPLES CONSIDERED")
        else:
            print(f"NUMBER OF SAMPLES: {nSamples}")
        nSamples *= self.nNodes
        dataset = preprocessing.normalize(randn(nSamples, self.nVars), norm = 'l2')
        response = randn(nSamples, 1)
        response[response >= 0.5] = 1
        response[response < 0.5] = 0
        self.dataset = dataset
        self.response = response

    def create_variables(self):
        self.model = ConcreteModel()
        self.model.x = Var(range(self.nVars))
        self.model.delta = Var(range(self.nVars), within = Binary)

    def create_objective(self):
        self.model.obj = Objective(expr = self._objective(), sense = minimize)

    def create_constraints(self, bound):
        self.model.limits = ConstraintList()
        for i in range(self.nVars):
            self.model.limits.add(self.model.x[i] <= bound * self.model.delta[i])
            self.model.limits.add(-bound * self.model.delta[i] <= self.model.x[i])
        self.model.limits.add(sum([self.model.delta[i] for i in range(self.nVars)]) <= self.nZeros)

    def _objective(self):
        m = self.dataset.shape[0]
        f = sum([- self.response[i][0] * log(self._logistic(self.dataset[i, :])) - (1 - self.response[i][0]) * log(
            1 - self._logistic(self.dataset[i, :])) for i in range(m)])
        return f

    def _logistic(self, sample):
        z = sum([self.model.x[i] * sample[i] for i in range(self.nVars)])

        return 1 / (1 + exp(-z))
