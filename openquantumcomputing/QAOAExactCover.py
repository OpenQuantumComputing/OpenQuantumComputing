from qiskit import *
import numpy as np
import math

from qiskit.circuit import Parameter

from openquantumcomputing.QAOABase import QAOABase

class QAOAExactCover(QAOABase):

    def __init__(self, params=None):
        super().__init__(params=params)
        self.FR = self.params.get('FR', None)
        self.CR = self.params.get('CR', None)
        self.mu = self.params.get('mu', 1)
        self.N_qubits = self.params.get('instances')

    def __exactCover(self, x):
        Cexact = np.sum((1 - (self.FR @ x))**2)
        return Cexact

    def cost(self, string):
        x = np.array(list(map(int,string)))
        c_e = self.__exactCover(x)


        if self.CR is None:
            return -c_e
        else:
            return - (self.CR@x + self.mu*c_e)

    def isFeasible(self, string, feasibleOnly=False):
        x = np.array(list(map(int,string)))
        c_e = self.__exactCover(x)
        if math.isclose(c_e, 0,abs_tol=1e-7):
            return True
        else:
            return False


    def create_cost_circuit(self, d, q):
        """
        Creates parameterized circuit corresponding to the cost function
        """
        
        self.gamma_params[d] = Parameter('gamma_' + str(d))
        usebarrier = self.params.get('usebarrier', False)
        if usebarrier:
            self.parameterized_circuit.barrier()
        
        F, R  = np.shape(self.FR)

        ### cost Hamiltonian
        for r in range(R):
            hr = self.mu * 0.5 * self.FR[:,r] @ (np.sum(self.FR,axis = 1) - 2)
            if not self.CR is None:
                hr += 0.5 * self.CR[r]


            if not math.isclose(hr, 0,abs_tol=1e-7):
                self.parameterized_circuit.rz( self.gamma_params[d] * hr, q[r])

            for r_ in range(r+1,R):
                Jrr_  = self.mu*0.5 * self.FR[:,r] @ self.FR[:,r_]

                if not math.isclose(Jrr_, 0,abs_tol=1e-7):
                    self.parameterized_circuit.cx(q[r], q[r_])
                    self.parameterized_circuit.rz(self.gamma_params[d] * Jrr_, q[r_])
                    self.parameterized_circuit.cx(q[r], q[r_])
        if usebarrier:
            self.parameterized_circuit.barrier()



    def create_mixer_circuit(self, d, q):
        self.beta_params[d] = Parameter('beta_'+str(d))
        q = QuantumRegister(self.N_qubits) 
        c = ClassicalRegister(self.N_qubits)

        self.mixer_circuit = QuantumCircuit(q, c)
        self.mixer_circuit.rx(-2 * self.beta_params[d], range(self.N_qubits))
        self.parameterized_circuit.compose(self.mixer_circuit, inplace = True)

        usebarrier = self.params.get('usebarrier', False)
        if usebarrier:
            self.parameterized_circuit.barrier()

