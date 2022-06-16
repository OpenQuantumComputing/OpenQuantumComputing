from qiskit import *
import numpy as np

from openquantumcomputing.QAOABase import QAOABase

class QAOAMaxKCutOnehot(QAOABase):

    def kBits_MaxKCut(self, k_cuts):
        return int(np.ceil(np.log2(k_cuts)))

    def binstringToLabels_MaxKCut(self, k_cuts,num_V,binstring):
        k_bits = self.kBits_MaxKCut(k_cuts)
        label_list = [int(binstring[j*k_bits:(j+1)*k_bits], 2) for j in range(num_V)]
        label_string = ''
        for label in label_list:
            label_string += str(label)
        return label_string

    def cost(self, string, params):
        G = params.get('G', None)
        k_cuts = params.get('k_cuts', None)
        num_V = G.number_of_nodes()

        C = 0
        labels = self.binstringToLabels_MaxKCut(k_cuts,num_V,string)
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
        k_bits = self.kBits_MaxKCut(k_cuts)
        if version==1:
            if k_cuts==2:
                Hij = np.array((-1, 1,
                                 1,-1,))
            elif k_cuts==3:
                Hij = np.array((-1, 1, 1, 1,
                                 1,-1, 1, 1,
                                 1, 1,-1,-1,
                                 1, 1,-1,-1))
            elif k_cuts==4:
                Hij = np.array((-1, 1, 1, 1,
                                 1,-1, 1, 1,
                                 1, 1,-1, 1,
                                 1, 1, 1,-1))
            elif k_cuts==5:
                Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                                 1,-1, 1, 1, 1, 1, 1, 1, 
                                 1, 1,-1, 1, 1, 1, 1, 1, 
                                 1, 1, 1,-1, 1, 1, 1, 1, 
                                 1, 1, 1, 1,-1,-1,-1,-1, 
                                 1, 1, 1, 1,-1,-1,-1,-1, 
                                 1, 1, 1, 1,-1,-1,-1,-1, 
                                 1, 1, 1, 1,-1,-1,-1,-1)) 
            elif k_cuts==6:
                Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                                 1,-1, 1, 1, 1, 1, 1, 1, 
                                 1, 1,-1, 1, 1, 1, 1, 1, 
                                 1, 1, 1,-1, 1, 1, 1, 1, 
                                 1, 1, 1, 1,-1, 1, 1, 1, 
                                 1, 1, 1, 1, 1,-1,-1,-1, 
                                 1, 1, 1, 1, 1,-1,-1,-1, 
                                 1, 1, 1, 1, 1,-1,-1,-1)) 
            elif k_cuts==7:
                Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                                 1,-1, 1, 1, 1, 1, 1, 1, 
                                 1, 1,-1, 1, 1, 1, 1, 1, 
                                 1, 1, 1,-1, 1, 1, 1, 1, 
                                 1, 1, 1, 1,-1, 1, 1, 1, 
                                 1, 1, 1, 1, 1,-1, 1, 1, 
                                 1, 1, 1, 1, 1, 1,-1,-1, 
                                 1, 1, 1, 1, 1, 1,-1,-1)) 
            elif k_cuts==8:
                Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                                 1,-1, 1, 1, 1, 1, 1, 1, 
                                 1, 1,-1, 1, 1, 1, 1, 1, 
                                 1, 1, 1,-1, 1, 1, 1, 1, 
                                 1, 1, 1, 1,-1, 1, 1, 1, 
                                 1, 1, 1, 1, 1,-1, 1, 1, 
                                 1, 1, 1, 1, 1, 1,-1, 1, 
                                 1, 1, 1, 1, 1, 1, 1,-1)) 
            else:
                raise Exception("Circuit creation for k=",k_cuts," not implemented for version 1 (hard coded).")

        # we need 2 auxillary qubits if k is not a power of two
        num_aux=0
        k_is_power_of_two = math.log(k_cuts, 2).is_integer()
        if version==2 and not k_is_power_of_two:
            num_aux=2
            ind_a1=num_V * k_bits + num_aux - 2
            ind_a2=num_V * k_bits + num_aux - 1

        q = QuantumRegister(num_V * k_bits + num_aux)
        c = ClassicalRegister(num_V * k_bits)
        circ = QuantumCircuit(q, c, name=name)
        circ.h(range(num_V * k_bits))

        if usebarrier:
            circ.barrier()
        for d in range(depth):
            gamma = angles[2 * d]
            beta = angles[2 * d + 1]
            if version==1:
                for edge in G.edges():
                    i = int(edge[0])
                    j = int(edge[1])
                    w = G[edge[0]][edge[1]]['weight']
                    wg = w * gamma
                    I = i * k_bits
                    J = j * k_bits
                    ind_Hij = [i_ for i_ in range(I, I+k_bits)]
                    for j_ in range(J,J+k_bits):
                        ind_Hij.append(j_)
                    U = np.diag(np.exp(-1j * (-wg) / 2 * Hij))
                    circ.unitary(U, ind_Hij, 'Hij('+"{:.2f}".format(wg)+")")
            else:
                if k_cuts == 2:
                    for edge in G.edges():
                        i = int(edge[0])
                        j = int(edge[1])
                        w = G[edge[0]][edge[1]]['weight']
                        wg = w * gamma
                        circ.cx(q[i], q[j])
                        circ.rz(wg, q[j])
                        circ.cx(q[i], q[j])
                        # this is an equivalent implementation:
                        #    circ.cu1(-2 * wg, i, j)
                        #    circ.u1(wg, i)
                        #    circ.u1(wg, j)
                        if usebarrier:
                            circ.barrier()
                elif k_is_power_of_two:
                    for edge in G.edges():
                        i = int(edge[0])
                        j = int(edge[1])
                        w = G[edge[0]][edge[1]]['weight']
                        wg = w * gamma
                        I = i * k_bits
                        J = j * k_bits
                        for k in range(k_bits):
                            circ.cx(I + k, J + k)
                            circ.x(J + k)
                        Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                        for k in reversed(range(k_bits)):
                            circ.x(J + k)
                            circ.cx(I + k, J + k)
                        if usebarrier:
                            circ.barrier()
                elif k_cuts == 3:
                    for edge in G.edges():
                        i = int(edge[0])
                        j = int(edge[1])
                        w = G[edge[0]][edge[1]]['weight']
                        wg = w * gamma
                        I = i * k_bits
                        J = j * k_bits

                        for k in range(k_bits):
                            circ.cx(I + k, J + k)
                            circ.x(J + k)
                        Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                        for k in reversed(range(k_bits)):
                            circ.x(J + k)
                            circ.cx(I + k, J + k)
                        if usebarrier:
                            circ.barrier()
                        circ.x(I)
                        circ.ccx(I,I+1,ind_a1)
                        circ.ccx(J,J+1,ind_a2)
                        Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                        circ.ccx(J,J+1,ind_a2)
                        circ.ccx(I,I+1,ind_a1)
                        circ.x(I)
                        if usebarrier:
                            circ.barrier()
                        circ.x(J)
                        circ.ccx(I,I+1,ind_a1)
                        circ.ccx(J,J+1,ind_a2)
                        Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                        circ.ccx(J,J+1,ind_a2)
                        circ.ccx(I,I+1,ind_a1)
                        circ.x(J)

                        if usebarrier:
                            circ.barrier()
                elif k_cuts >= 5 and k_cuts<=7:
                    tt={}
                    tt[0]=[False,False,False]
                    tt[1]=[True,False,False]
                    tt[2]=[False,True,False]
                    tt[3]=[True,True,False]
                    tt[4]=[False,False,True]
                    for edge in G.edges():
                        i = int(edge[0])
                        j = int(edge[1])
                        w = G[edge[0]][edge[1]]['weight']
                        wg = w * gamma
                        I = i * k_bits
                        J = j * k_bits

                        for k in range(k_bits):
                            circ.cx(I + k, J + k)
                            circ.x(J + k)
                        Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                        for k in reversed(range(k_bits)):
                            circ.x(J + k)
                            circ.cx(I + k, J + k)

                        if usebarrier:
                            circ.barrier()

                        n=2**k_bits-k_cuts+1
                        for fi in range(n):
                            for fj in range(n):
                                if fi==fj:
                                    continue
                                itt=-1
                                for apply in tt[fi]:
                                    itt+=1
                                    if apply:
                                        circ.x(I+itt)
                                jtt=-1
                                for apply in tt[fj]:
                                    jtt+=1
                                    if apply:
                                        circ.x(J+jtt)
                                circ.mcx([I,I+1,I+2],ind_a1)
                                circ.mcx([J,J+1,J+2],ind_a2)
                                Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                                circ.mcx([J,J+1,J+2],ind_a2)
                                circ.mcx([I,I+1,I+2],ind_a1)
                                itt=-1
                                for apply in tt[fi]:
                                    itt+=1
                                    if apply:
                                        circ.x(I+itt)
                                jtt=-1
                                for apply in tt[fj]:
                                    jtt+=1
                                    if apply:
                                        circ.x(J+jtt)
                                if usebarrier:
                                    circ.barrier()

                        if usebarrier:
                            circ.barrier()
                else:
                    raise Exception("Circuit creation for k=",k_cuts," not implemented for version 2 (decomposed).")

            circ.rx(-2 * beta, range(num_V * k_bits))
            if usebarrier:
                circ.barrier()
        if version == 2 and not k_is_power_of_two:
            circ.measure(q[:-2], c)
        else:
            circ.measure(q, c)
        return circ
