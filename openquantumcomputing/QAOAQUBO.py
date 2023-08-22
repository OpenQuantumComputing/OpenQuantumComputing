from qiskit import *
import numpy as np
import math

from qiskit.circuit import Parameter

from openquantumcomputing.QAOABase import QAOABase


class QAOAQUBO(QAOABase):
    def __init__(self, params=None):
        """
        init function that initializes QUBO.
        The aim is to solve the problem
        min x^T Q x + c^T x + b
        for n-dimensional binary variable x

        :param params: additional parameters
        """
        super().__init__(params=params)

        self.parameterized = False
        self.QUBO_Q = None
        self.QUBO_c = None
        self.QUBO_b = None

        self.lower_triangular_Q = False

    def _init_QUBO(self, Q=None, c=None, b=None):
        """
        Implements the mapping from the parameters in params to the QUBO problem.
        Is expected to be called by the child class.
        """
        assert type(Q) is np.ndarray, "Q needs to be a numpy ndarray, but is " + str(
            type(Q)
        )
        assert (
            Q.ndim == 2
        ), "Q needs to be a 2-dimensional numpy ndarray, but has dim " + str(Q.ndim)
        assert Q.shape[0] == Q.shape[1], "Q needs to be a square matrix, but is " + str(
            Q.shape
        )
        n = Q.shape[0]

        # Check if Q is lower triangular
        self.lower_triangular_Q = np.allclose(Q, np.tril(Q))

        self.QUBO_Q = Q

        if c is None:
            c = np.zeros(n)
        assert type(c) is np.ndarray, "c needs to be a numpy ndarray, but is " + str(
            type(c)
        )
        assert (
            c.ndim == 1
        ), "c needs to be a 1-dimensional numpy ndarray, but has dim " + str(Q.ndim)
        assert c.shape[0] == n, (
            "c is of size "
            + str(c.shape[0])
            + " but should be compatible size to Q, meaning "
            + str(n)
        )
        self.QUBO_c = c

        if b is None:
            b = 0.0
        assert np.isscalar(b), "b is expected to be scalar, but is " + str(b)
        self.QUBO_b = b

    def cost(self, string):
        x = np.array(list(map(int, string)))
        return -(x.T @ self.QUBO_Q @ x + self.QUBO_c.T @ x + self.QUBO_b)

    def create_cost_circuit(self):
        if self.lower_triangular_Q:
            self._createParameterizedCostCircuitTril()
        else:
            # Not lower triangular Q matrix
            raise NotImplementedError

    def _createParameterizedCostCircuitTril(self):
        """
        Creates a parameterized circuit of the triangularized QUBO problem.
        """
        q = QuantumRegister(self.N_qubits)
        self.cost_circuit = QuantumCircuit(q)
        cost_param = Parameter("x_gamma")

        usebarrier = self.params.get("usebarrier", False)
        if usebarrier:
            self.cost_circuit.barrier()

        ### cost Hamiltonian
        for i in range(self.N_qubits):
            w_i = 0.5 * (self.QUBO_c[i] + np.sum(self.QUBO_Q[:, i]))

            if not math.isclose(w_i, 0, abs_tol=1e-7):
                self.cost_circuit.rz(cost_param * w_i, q[i])

            for j in range(i + 1, self.N_qubits):
                w_ij = 0.25 * self.QUBO_Q[j][i]

                if not math.isclose(w_ij, 0, abs_tol=1e-7):
                    self.cost_circuit.cx(q[i], q[j])
                    self.cost_circuit.rz(cost_param * w_ij, q[j])
                    self.cost_circuit.cx(q[i], q[j])
        if usebarrier:
            self.cost_circuit.barrier()
