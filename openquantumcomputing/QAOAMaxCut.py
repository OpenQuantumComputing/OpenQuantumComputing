from qiskit import *
from qiskit.circuit import Parameter
import numpy as np

from openquantumcomputing.QAOABase import QAOABase

class QAOAMaxCut(QAOABase):

    def __init__(self, params=None):
        super().__init__(params=params)

        self.G = self.params.get('G', None)
        self.N_qubits = self.G.number_of_nodes()



    def cost(self, string):

        C = 0
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            if string[i] != string[j]:
                w =self. G[edge[0]][edge[1]]['weight']
                C += w
        return C
    
    def create_cost_circuit(self, d, q):
        """
        Adds a parameterized circuit for the cost part to the member variable self.parameteried_circuit
        and a parameter to the parameter list self.gamma_params
        """
        self.gamma_params[d] = Parameter('gamma_' + str(d))
        usebarrier = self.params.get('usebarrier', False)
        if usebarrier:
            self.parameterized_circuit.barrier()

        ### cost Hamiltonian
        for edge in G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = self.G[edge[0]][edge[1]]['weight']
            wg = w * self.gamma_params[d]
            self.parameterized_circuit.cx(q[i], q[j])
            self.parameterized_circuit.rz(wg, q[j])
            self.parameterized_circuit.cx(q[i], q[j])
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




   
   
