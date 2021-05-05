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
        self.model.obj = Objective(expr = self._objective(self.dataset, self.response), sense = minimize)

    def create_constraints(self, bound):
        self.model.limits = ConstraintList()
        for i in range(self.nVars):
            self.model.limits.add(self.model.x[i] <= bound * self.model.delta[i])
            self.model.limits.add(-bound * self.model.delta[i] <= self.model.x[i])
        self.model.limits.add(sum([self.model.delta[i] for i in range(self.nVars)]) <= self.nZeros)

    def solve(self):

        solver = SolverFactory('gams')
        # io_options=dict(add_options=['reslim=100;'])
        results = solver.solve(self.model, solver = 'bonmin', tee = False, keepfiles = False, add_options=['GAMS_MODEL.reslim = 0.1;'])
        print(results)
        print([value(self.model.delta[i]) for i in range(self.nVars)])
        print([value(self.model.x[i]) for i in range(self.nVars)])
        return results

    def _objective(self, dataset = None, response = None):
        m = dataset.shape[0]
        f = sum([- response[i][0] * log(self._logistic(dataset[i, :])) - (1 - response[i][0]) * log(
            1 - self._logistic(dataset[i, :])) for i in range(m)])
        return f

    def _logistic(self, sample):
        z = sum([self.model.x[i] * sample[i] for i in range(self.nVars)])

        return 1 / (1 + exp(-z))
