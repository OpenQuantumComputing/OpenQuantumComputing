from qiskit import *
import numpy as np
import math

from openquantumcomputing.QAOABase import QAOABase

class QAOAExactCover(QAOABase):


    def __exactCover(self, x):
        FR = self.params.get('FR', None)
        Cexact = np.sum((1 - (FR @ x))**2)
        return Cexact

    def cost(self, string):
        x = np.array(list(map(int,string)))
        c_e = self.__exactCover(x)
        CR = self.params.get('CR', None)
        mu = self.params.get('mu', 1)

        if CR is None:
            return -c_e
        else:
            return - (CR@x + mu*c_e)

    def isFeasible(self, string, feasibleOnly=False):
        x = np.array(list(map(int,string)))
        c_e = self.__exactCover(x)
        if math.isclose(c_e, 0,abs_tol=1e-7):
            return True
        else:
            return False

    def createCircuit(self, angles, depth):
        FR = self.params.get('FR', None)
        CR = self.params.get('CR', None)
        mu = self.params.get('mu', 1)
        usebarrier = self.params.get('usebarrier', False)

        F, R  = np.shape(FR)

        q = QuantumRegister(R)
        c = ClassicalRegister(R)
        circ = QuantumCircuit(q, c)

        ### initial state
        circ.h(range(R))

        if usebarrier:
            circ.barrier()
        for d in range(depth):
            gamma = angles[2 * d]
            beta = angles[2 * d + 1]
            ### cost Hamiltonian
            for r in range(R):
                hr = mu * 0.5 * FR[:,r] @ (np.sum(FR,axis = 1) - 2)
                if not CR is None:
                    hr += 0.5 * CR[r]


                if not math.isclose(hr, 0,abs_tol=1e-7):
                    circ.rz( gamma * hr, q[r])

                for r_ in range(r+1,R):
                    Jrr_  = mu*0.5 * FR[:,r] @ FR[:,r_]

                    if not math.isclose(Jrr_, 0,abs_tol=1e-7):
                        circ.cx(q[r], q[r_])
                        circ.rz(gamma * Jrr_, q[r_])
                        circ.cx(q[r], q[r_])
            if usebarrier:
                circ.barrier()
            ### mixer Hamiltonian
            circ.rx(-2 * beta, range(R))
            if usebarrier:
                circ.barrier()
        circ.measure(q, c)
        return circ
