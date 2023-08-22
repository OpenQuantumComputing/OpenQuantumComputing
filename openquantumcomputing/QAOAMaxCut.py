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
    
    def create_cost_circuit(self):
        """
        Adds a parameterized circuit for the cost part to the member variable self.parameteried_circuit
        and a parameter to the parameter list self.gamma_params
        """
        q = QuantumRegister(self.N_qubits) 
        self.cost_circuit = QuantumCircuit(q)
        cost_param= Parameter('x_gamma')
        usebarrier = self.params.get('usebarrier', False)

        if usebarrier:
            self.cost_circuit.barrier()

        ### cost Hamiltonian
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = self.G[edge[0]][edge[1]]['weight']
            wg = w * cost_param
            self.cost_circuit.cx(q[i], q[j])
            self.cost_circuit.rz(wg, q[j])
            self.cost_circuit.cx(q[i], q[j])
            if usebarrier:
                self.cost_circuit.barrier()
                




   
   
