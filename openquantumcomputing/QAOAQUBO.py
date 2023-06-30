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
        super(QAOAQUBO, self).__init__(params=params)

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
        assert(type(Q) is np.ndarray), "Q needs to be a numpy ndarray, but is "+str(type(Q))
        assert(Q.ndim == 2), "Q needs to be a 2-dimensional numpy ndarray, but has dim "+str(Q.ndim)
        assert(Q.shape[0] == Q.shape[1]), "Q needs to be a square matrix, but is "+str(Q.shape)
        n = Q.shape[0]

        # Check if Q is lower triangular 
        self.lower_triangular_Q = np.allclose(Q, np.tril(Q))

        self.QUBO_Q = Q

        if c is None:
            c = np.zeros(n)
        assert(type(c) is np.ndarray), "c needs to be a numpy ndarray, but is "+str(type(c))
        assert(c.ndim == 1), "c needs to be a 1-dimensional numpy ndarray, but has dim "+str(Q.ndim)
        assert(c.shape[0] == n), "c is of size "+str(c.shape[0])+" but should be compatible size to Q, meaning "+str(n)
        self.QUBO_c = c

        if b is None:
            b = 0.0
        assert(np.isscalar(b)), "b is expected to be scalar, but is "+str(b)
        self.QUBO_b = b

        
    def cost(self, string):
        x = np.array(list(map(int, string)))
        return - (x.T@self.QUBO_Q@x + self.QUBO_c.T@x + self.QUBO_b)

    def createCircuit(self, angles, depth):
        #print("depth: ", depth)
        if self.lower_triangular_Q:
            if self.use_parameterized_circuit:
                if self.current_circuit_depth != depth:
                    self._createParameterizedCircuitTril( depth)
                return self._applyParameters(angles, depth)
            return self._createCircuitTril(angles, depth)
        
        else:
            # Full Q-matrix
            if self.use_parameterized_circuit:
                raise NotImplementedError
            return self._createCircuitFull(angles, depth)
      

    def _createCircuitTril(self, angles, depth):
        """
        Creates a quantum circuit of the triangularized QUBO problem for the given angles.
        """
        
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
    
    def _createParameterizedCircuitTril(self, depth):
        """
        Creates a parameterized circuit of the triangularized QUBO problem.
        """
        self.gamma_params = [None]*depth
        self.beta_params = [None]*depth
        for d in range(depth):
            self.gamma_params[d] = Parameter('gamma_'+ str(d))
            self.beta_params[d]  = Parameter('beta_' + str(d))

        usebarrier = self.params.get('usebarrier', False)

        q = QuantumRegister(self.N_assets)
        c = ClassicalRegister(self.N_assets)
        self.parameterized_circuit = QuantumCircuit(q, c)

        ### initial state
        self.parameterized_circuit.h(range(self.N_assets))

        #self.num_created_circuits[depth-1] += 1

        if usebarrier:
            self.parameterized_circuit.barrier()
        for d in range(depth):
            ### cost Hamiltonian
            for i in range(self.N_assets):
                w_i = 0.5 * (self.QUBO_c[i] + np.sum(self.QUBO_Q[:, i]))
                

                if not math.isclose(w_i, 0,abs_tol=1e-7):
                    self.parameterized_circuit.rz( self.gamma_params[d] * w_i, q[i])

                for j in range(i+1, self.N_assets):
                    w_ij = 0.25*self.QUBO_Q[j][i]

                    if not math.isclose(w_ij, 0,abs_tol=1e-7):
                        self.parameterized_circuit.cx(q[i], q[j])
                        self.parameterized_circuit.rz(self.gamma_params[d] * w_ij, q[j])
                        self.parameterized_circuit.cx(q[i], q[j])
                if usebarrier:
                    self.parameterized_circuit.barrier()
            ### mixer Hamiltonian
            self.parameterized_circuit.rx(-2 * self.beta_params[d], range(self.N_assets))
            if usebarrier:
                self.parameterized_circuit.barrier()
        self.parameterized_circuit.measure(q, c)
        self.current_circuit_depth = depth



    def _createCircuitFull(self, angles, depth):
        raise NotImplementedError 
        ## TODO: Implement so that this class can be applied to general QUBO problems, and not just
        # thos with symmetric matrices in the quadratic term.
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

