from qiskit import *
import numpy as np
import math

import qiskit.quantum_info as qi
from qiskit.circuit.library import PauliEvolutionGate


from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
#run: pip install openquantumcomputing
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/olaib/QuantumComputing/OpenQuantumComputing')
sys.path.append('/Users/olaib/QuantumComputing/OpenQuantumComputing_private')
from openquantumcomputing2.Mixer import *
from openquantumcomputing.QAOABase import QAOABase


class QAOAConstrained_design_mixer(QAOABase):

    def __init__(self, params=None):
        """
        init function that initializes QUBO.
        The aim is to solve the problem
        min x^T Q x + c^T x + b 
        for n-dimensional binary variable x

        :param params: additional parameters
        """
        super(QAOAConstrained_design_mixer, self).__init__(params=params)
        self.QUBO_Q = None 
        self.QUBO_c = None 
        self.QUBO_b = None
        self.B = []
        self.best_mixer_terms = []
        self.mixer_circuit = None
        self.reduced = True #should be taken in with params

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
        #print("standard cost function called")
        x = np.array(list(map(int, string)))
        #print("Return value: ",  - (x.T@self.QUBO_Q@x + self.QUBO_c.T@x + self.QUBO_b))
        return - (x.T@self.QUBO_Q@x + self.QUBO_c.T@x + self.QUBO_b)


    def createCircuit(self, angles, depth):
        if self.lower_triangular_Q:
            if self.use_parameterized_circuit:
                if self.current_circuit_depth != depth:
                    self._createParameterizedCircuitTril( depth)
                return self._applyParameters(angles, depth)
            return self._createCircuitTril(angles, depth)
        else:
            #Full Q-matrix
            if self.use_parameterized_circuit:
                raise NotImplementedError
            return self._createCircuitFull(angles, depth)
      
      

    def _createCircuitTril(self, angles, depth):

        usebarrier = self.params.get('usebarrier', False)

        q = QuantumRegister(self.N_assets)
        c = ClassicalRegister(self.N_assets)
        circ = QuantumCircuit(q, c)

        ### initial state     ++++++What should this be now+++++++????
        #circ.h(range(self.N_assets)) For standard (unconstrained) mixer
        self.setToInitialState(circ, q)
        self.computeBestMixer()
        

        if usebarrier:
            circ.barrier()
        for d in range(depth):
            gamma = angles[2 * d]
            beta_val = angles[2 * d + 1]
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
            #circ.rx(-2 * beta, range(self.N_assets))  Standard mixer
            circ = self.applyBestMixer(circ, beta_val, q)   #Adding the best mixer foud by Franz to the circuit circ
            
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
        self.setToInitialState(self.parameterized_circuit, q)

        self.computeBestMixer()
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

            self.parameterized_circuit = self.applyBestMixer(self.parameterized_circuit, self.beta_params[d], q)
            if usebarrier:
                self.parameterized_circuit.barrier()
        self.parameterized_circuit.measure(q, c)
        self.current_circuit_depth = depth




    def _createCircuitFull(self, angles, depth):
        raise NotImplementedError 
    
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
    
    def applyBestMixer(self, circuit, beta, q):    
        if not self.best_mixer_terms:
            #self.computeBestMixer(beta)
            #should not go in here
            raise NotImplementedError

        if self.use_parameterized_circuit:
            # beta is a parameter
            c = self.mixer_circuit.assign_parameters({self.mixer_circuit.parameters[0]: beta}, inplace = False)
            c_return = circuit.compose(c, inplace = False)
            return c_return
        else:
            #beta is a float

            for term in self.best_mixer_terms:
                circuit.append(PauliEvolutionGate(qi.Pauli(term.P), time = np.real(term.scalar)*beta), q)
            return circuit




        """
        for term in self.best_mixer_terms:
            circuit.barrier()
            indicies_of_X = [i for i in range(len(term)) if term[i] == 'X']
            
            if (len(indicies_of_X) == 1):
                #If pauli string only contains one X operator, an X-rotation is applied to that qubit
                circuit.rx(-2*beta, indicies_of_X[0])
            else:
                #If pauli string contains more than one X operator we apply the "CNOT cascade" for the relevant qubits (see paper)
                circuit.h(indicies_of_X) #apply hadamard gate to all relevant qubits
                
                for i in range(len(indicies_of_X)-1):
                    circuit.cx(indicies_of_X[i], indicies_of_X[i+1])
            
                circuit.rz(2*beta, qubit_register[indicies_of_X[-1]])

                for i in reversed(range(1, len(indicies_of_X))):
                    circuit.cx(indicies_of_X[i-1], indicies_of_X[i])
                circuit.h(indicies_of_X)

        """

    def computeBestMixer(self):
        if not self.B:
            self.computeFeasibleSubspace()
        if not self.best_mixer_terms and self.use_parameterized_circuit:
                
            print("Its now computing the best mixer, parametrized by Havard")
            m = Mixer(self.B, sort = True)
            m.compute_commuting_pairs()
            m.compute_family_of_graphs()
            m.get_best_mixer_commuting_graphs(reduced = self.reduced)
            self.mixer_circuit, self.best_mixer_terms = m.compute_parametrized_circuit(self.reduced)
        elif not self.best_mixer_terms and not self.use_parameterized_circuit:
            print("Its now computing the best mixer, not parametrized")
            m = Mixer(self.B, sort = True)
            m.compute_commuting_pairs()
            m.compute_family_of_graphs()
            m.get_best_mixer_commuting_graphs(reduced = self.reduced)
            self.best_mixer_terms = m.compute_mixer_terms(self.reduced)
        else:
            pass

        



    def computeFeasibleSubspace(self):
        """
        To be implemented by a child class,
        where a constraint is included and the feasbible subspace corresponding
        to this constraint is computed using this function
        """
        raise NotImplementedError
    

    def setToInitialState(self, circuit, quantum_register):
        #set to ground state of mixer hamilton??
        if not self.B:
            self.computeFeasibleSubspace()
                # initial state
        ampl_vec = np.zeros(2 ** len(self.B[0]))
        ampl = 1 / np.sqrt(len(self.B))
        for state in self.B:
            ampl_vec[int(state, 2)] = ampl
        
        circuit.initialize(ampl_vec, quantum_register)