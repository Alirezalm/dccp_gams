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


class SparseQCQP(RandDCCP):

    obj_data = {
        'hess': None,
        'grad': None
    }
    constr_data = {
        'hess': None,
        'grad': None,
        'const': - rand()
    }

    def generate_data(self):
        self.nVars *= self.nNodes
        num_qc = 1
        for i in range(num_qc + 1):
            _hess = preprocessing.normalize(randn(self.nVars, self.nVars), norm = 'l2')
            _hess = 0.5 * (_hess.T + _hess)
            diag_mat = (1 + rand()) * eye(self.nVars)
            hess = _hess.T @ _hess + diag_mat
            grad = rand(self.nVars, 1)
            if i == 0:
                self.obj_data['hess'] = hess
                self.obj_data['grad'] = grad
            else:
                self.constr_data['hess'] = hess
                self.constr_data['grad'] = grad

    def create_variables(self):

        self.model = ConcreteModel()
        self.model.x = Var(range(self.nVars))
        self.model.delta = Var(range(self.nVars), within = Binary)

    def create_objective(self):

        self.model.obj = Objective(expr = self._objective(), sense = minimize)

    def _objective(self):
        z = []
        for i in range(self.nVars):
            z.append(sum([self.model.x[j] * self.obj_data['hess'][j, i] for j in range(self.nVars)]))

        obj = sum([z[i] * self.model.x[i] for i in range(self.nVars)]) + sum(
            [self.model.x[i] * self.obj_data['grad'][i][0] for i in range(self.nVars)])
        return sum([ self.model.x[i]**2 for i in range(self.nVars)])

    def create_constraints(self, bound):
        self.model.limits = ConstraintList()
        for i in range(self.nVars):
            self.model.limits.add(self.model.x[i] <= bound * self.model.delta[i])
            self.model.limits.add(-bound * self.model.delta[i] <= self.model.x[i])
        self.model.limits.add(sum([self.model.delta[i] for i in range(self.nVars)]) <= self.nZeros)

        # z = []
        # for i in range(self.nVars):
        #     z.append(sum([self.model.x[j] * self.constr_data['hess'][j, i] for j in range(self.nVars)]))
        # const = sum([z[i] * self.model.x[i] for i in range(self.nVars)]) + sum(
        #     [self.model.x[i] * self.constr_data['grad'][i][0] for i in range(self.nVars)])
        # self.model.limits.add(const + self.constr_data['const'] <= 0)
