from qiskit import *
import numpy as np
import math

from openquantumcomputing.QAOABase import QAOABase

class QAOAPortfolioOptimization(QAOABase):

    def __init__(self, params=None):
        super(QAOAPortfolioOptimization, self).__init__(params=params)

        self.__checkParams()
        self.risk = params.get("risk")
        self.budget = params.get("budget")
        self.cov_matrix = params.get("cov_matrix")
        self.exp_return = params.get("exp_return")
        self.penalty = params.get("penalty", 0.0)
        self.N_assets = len(self.exp_return)

        # Reformulated as a QUBO
        # min x^T Q x + c^T x + b
        self.QUBO_Q = self.risk * np.tril(self.cov_matrix + np.tril(self.cov_matrix, k=-1)) \
                        + self.penalty*(np.eye(self.N_assets) + 2* np.tril(np.ones((self.N_assets, self.N_assets)), k=-1))
        self.QUBO_c = - self.exp_return - (2*self.penalty*self.budget*np.ones_like(self.exp_return))
        self.QUBO_b = self.penalty*self.budget*self.budget

    def __checkParams(self):
        # we require the following params:
        for key in ["risk", "budget", "cov_matrix", "exp_return"]:
            assert(key in self.params), "missing required parameter " + key
         

    def __str2np(self, s):
        x = np.array(list(map(int, s)))
        assert(len(x) == len(self.params.get("exp_return"))), \
            "bitstring  " + s + " of wrong size. Expected " + str(len(self.params.get("exp_return"))) + " but got " + str(len(x))
        return x

    def cost(self, string, penalize=True):
        
        risk       = self.params.get("risk")
        budget     = self.params.get("budget")
        cov_matrix = self.params.get("cov_matrix")
        exp_return = self.params.get("exp_return")
        penalty    = self.params.get("penalty", 0.0)

        x = np.array(list(map(int,string)))        
        cost = risk* (x.T@cov_matrix@x) - exp_return.T@x
        if penalize:
            cost += penalty * (x.sum() - budget)**2

        return -cost
    
    def cost_QUBO(self, s):
        x = np.array(list(map(int, s)))
        return x.T@self.QUBO_Q@x + self.QUBO_c.T@x + self.QUBO_b

    def costAlt(self, string, penalize=True):
        risk       = self.params.get("risk")
        budget     = self.params.get("budget")
        cov_matrix = self.params.get("cov_matrix")
        exp_return = self.params.get("exp_return")
        penalty    = self.params.get("penalty", 0.0)

        
        x = np.array(list(map(int,string)))        
        cost = risk* (x.T@cov_matrix@x) - exp_return.T@x
        if penalize:
            cost += penalty * (x.sum() - budget)**2

        return cost
        

    def isFeasible(self, string, feasibleOnly=False):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.params.get("budget")
        if math.isclose(constraint, 0,abs_tol=1e-7):
            return True
        else:
            return False

    def createCircuit(self, angles, depth):
        enum_circuit = self.params.get("circuit", 0)
        if enum_circuit == 0:
            return self._createCircuitTril(angles, depth)
        #elif enum_circuit == 1:
        #    return self._createCircuitFull(angles, depth)
        else:
            raise Exception("Circuit creation for type " + str(enum_circuit) + " not implemented.")


    def _createCircuitTril(self, angles, depth):
        
        usebarrier = self.params.get('usebarrier', False)

        q = QuantumRegister(self.N_assets)
        c = ClassicalRegister(self.N_assets)
        circ = QuantumCircuit(q, c)

        ### initial state
        circ.h(range(self.N_assets))

        if usebarrier:
            circ.barrier()
        for d in range(depth):
            gamma = angles[2 * d]
            beta = angles[2 * d + 1]
            ### cost Hamiltonian
            for i in range(self.N_assets):
                w_i = 0.5 * (self.QUBO_c[i] + np.sum(self.QUBO_Q[:, i]))
                

                if not math.isclose(w_i, 0,abs_tol=1e-7):
                    circ.rz( gamma * w_i, q[i])

                for j in range(i+1, self.N_assets):
                    w_ij = 0.25*self.QUBO_Q[j][i]

                    if not math.isclose(w_ij, 0,abs_tol=1e-7):
                        circ.cx(q[i], q[j])
                        circ.rz(gamma * w_ij, q[j])
                        circ.cx(q[i], q[j])
                if usebarrier:
                    circ.barrier()
            ### mixer Hamiltonian
            circ.rx(-2 * beta, range(self.N_assets))
            if usebarrier:
                circ.barrier()
        circ.measure(q, c)
        return circ


    def _createCircuitFull(self, angles, depth):
        
        usebarrier = self.params.get('usebarrier', False)

        q = QuantumRegister(self.N_assets)
        c = ClassicalRegister(self.N_assets)
        circ = QuantumCircuit(q, c)

        ### initial state
        circ.h(range(self.N_assets))

        if usebarrier:
            circ.barrier()
        for d in range(depth):
            gamma = angles[2 * d]
            beta = angles[2 * d + 1]
            ### cost Hamiltonian
            for i in range(self.N_assets):
                w_i = 0.5 * (self.QUBO_c[i] + np.sum(self.QUBO_Q[:, i]))
                

                if not math.isclose(w_i, 0,abs_tol=1e-7):
                    circ.rz( gamma * w_i, q[i])

                for j in range(i+1, self.N_assets):
                    w_ij = 0.25*self.QUBO_Q[j][i]

                    if not math.isclose(w_ij, 0,abs_tol=1e-7):
                        circ.cx(q[i], q[j])
                        circ.rz(gamma * w_ij, q[j])
                        circ.cx(q[i], q[j])
                if usebarrier:
                    circ.barrier()
            ### mixer Hamiltonian
            circ.rx(-2 * beta, range(self.N_assets))
            if usebarrier:
                circ.barrier()
        circ.measure(q, c)
        return circ
