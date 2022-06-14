from qiskit import *
import numpy as np

from openquantumcomputing.QAOABase import QAOABase

class QAOAMaxCut(QAOABase):

    def cost(self, string, params):
        G = params.get('G', None)
        C = 0
        for edge in G.edges():
            i = int(edge[0])
            j = int(edge[1])
            if string[i] != string[j]:
                w = G[edge[0]][edge[1]]['weight']
                C += w
        return C

    def createCircuit(self, angles, depth, params={}):
        G = params.get('G', None)
        usebarrier = params.get('usebarrier', False)
        name= params.get('name', "")

        num_V = G.number_of_nodes()

        q = QuantumRegister(num_V)
        c = ClassicalRegister(num_V)
        circ = QuantumCircuit(q, c, name=name)

        ### initial state
        circ.h(range(num_V))

        if usebarrier:
            circ.barrier()
        for d in range(depth):
            gamma = angles[2 * d]
            beta = angles[2 * d + 1]
            ### cost Hamiltonian
            for edge in G.edges():
                i = int(edge[0])
                j = int(edge[1])
                w = G[edge[0]][edge[1]]['weight']
                wg = w * gamma
                circ.cx(q[i], q[j])
                circ.rz(wg, q[j])
                circ.cx(q[i], q[j])
                if usebarrier:
                    circ.barrier()
            ### mixer Hamiltonian
            circ.rx(-2 * beta, range(num_V))
            if usebarrier:
                circ.barrier()
        circ.measure(q, c)
        return circ
