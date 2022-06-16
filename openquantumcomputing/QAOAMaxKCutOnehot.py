from qiskit import *
import numpy as np

from openquantumcomputing.QAOABase import QAOABase

class QAOAMaxKCutOnehot(QAOABase):

    def validcoloring_onehot(self, s):
        num_ones=0
        for i in range(len(s)):
            num_ones+=int(s[i])
            if num_ones>1:
                break
        val = True
        if num_ones!=1:
            val = False
        return val


    def validstring_onehot(self, s,num_V):
        if len(s)%num_V!=0:
            raise Exception("inconsistent lenght")
        l=int(len(s)/num_V)
        vale = True
        for i in range(num_V):
            ss=s[i*l:i*l+l]
            val=self.validcoloring_onehot(ss)
            #print(ss,val)
            if not val:
                break
        return val


    def cost(self, string, params):
        G = params.get('G', None)
        k_cuts = params.get('k_cuts', None)
        num_V = G.number_of_nodes()

        C = 0
        if self.validstring_onehot(string, num_V):
            labels = binstringToLabels_MaxKCut_onehot(string, num_V, k_cuts)
            for edge in G.edges():
                i = int(edge[0])
                j = int(edge[1])
                li=min(k_cuts-1,int(labels[i]))## e.g. for k_cuts=3, labels 2 and 3 should be equal
                lj=min(k_cuts-1,int(labels[j]))## e.g. for k_cuts=3, labels 2 and 3 should be equal
                if li != lj:
                    w = G[edge[0]][edge[1]]['weight']
                    C += w
        return C

    def createCircuit(self, angles, depth, params={}):
        G = params.get('G', None)
        k_cuts = params.get('k_cuts', None)
        alpha = params.get('alpha', None)
        version = params.get('version', 2)
        usebarrier = params.get('usebarrier', False)
        name= params.get('name', "")

        num_V = G.number_of_nodes()

        num_qubits = num_V * k_cuts

        q = QuantumRegister(num_qubits)
        c = ClassicalRegister(num_qubits)
        circ = QuantumCircuit(q, c, name=name)
        if version==1:
            circ.h(range(num_qubits))
        else:
            for v in range(num_V):
                I = v*k_cuts
                Wn(circ, [i for i in range(I, I+k_cuts)])
                #circ.initialize(W, [q[i] for i in range(I, I+k_cuts)])

        if usebarrier:
            circ.barrier()
        for d in range(depth):
            gamma = angles[2 * d]
            beta = angles[2 * d + 1]
            # the objective Hamiltonian
            for edge in G.edges():
                i = int(edge[0])
                j = int(edge[1])
                w = G[edge[0]][edge[1]]['weight']
                wg = w * gamma
                I = k_cuts * i
                J = k_cuts * j
                for k in range(k_cuts):
                    circ.cx(q[I+k], q[J+k])
                    circ.rz(wg, q[J+k])
                    circ.cx(q[I+k], q[J+k])
                if usebarrier:
                    circ.barrier()
            # the penalty Hamiltonian
            if alpha != None:
                for v in range(num_V):
                    I = v*k_cuts
                    for i in range(k_cuts):
                        for j in range(i+1,k_cuts):
                            circ.cx(q[I+i], q[I+j])
                            circ.rz(gamma*alpha, q[I+j])
                            #circ.rz(alpha, q[I+j])
                            circ.cx(q[I+i], q[I+j])
                    if usebarrier:
                        circ.barrier()
            if version==1:
                circ.rx(-2 * beta, range(num_qubits))
                if usebarrier:
                    circ.barrier()
            else:
                for v in range(num_V):
                    I = v*k_cuts
                    ## odd
                    for i in range(0,k_cuts-1,2):
                        circ.rxx(-2 * beta, q[I+i], q[I+i+1])
                        circ.ryy(-2 * beta, q[I+i], q[I+i+1])
                    ## even
                    for i in range(1,k_cuts,2):
                        circ.rxx(-2 * beta, q[I+i], q[I+(i+1)%k_cuts])
                        circ.ryy(-2 * beta, q[I+i], q[I+(i+1)%k_cuts])
                    # final
                    if k_cuts%2==1:
                        circ.rxx(-2 * beta, q[I+k_cuts-1], q[I])
                        circ.ryy(-2 * beta, q[I+k_cuts-1], q[I])
                    if usebarrier:
                        circ.barrier()

        circ.measure(q, c)
        return circ
