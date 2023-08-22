from qiskit import *
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import XXPlusYYGate
import numpy as np
import math
import itertools

# from openquantumcomputing.QAOAQUBO import QAOAQUBO

import sys

# caution: path[0] is reserved for script path (or '' in REPL)

from openquantumcomputing.QAOAConstrainedQUBO import QAOAConstrainedQUBO
from openquantumcomputing.PauliString import PauliString


class QAOAKhot(QAOAConstrainedQUBO):
    def __init__(self, params=None):
        super().__init__(params=params)

        self.k = None  # Number of ones in feasible strings. Must be initialized by a child class

    def __str2np(self, s):
        x = np.array(list(map(int, s)))
        assert len(x) == self.N_qubits, (
            "bitstring  "
            + s
            + " of wrong size. Expected "
            + str(self.N_qubits)
            + " but got "
            + str(len(x))
        )
        return x

    def isFeasible(self, string, feasibleOnly=False):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.k
        if math.isclose(constraint, 0, abs_tol=1e-7):
            return True
        else:
            return False

    def create_mixer_circuit(self):
        # Overrides this function of QAOAConstrainedQUBO for the k-hot problem where structure of mixer is known

        q = QuantumRegister(self.N_qubits)
        self.mixer_circuit = QuantumCircuit(q)
        self.best_mixer_terms, self.logical_X_operators = self.__XYMixerTerms()

        Beta = Parameter("x_beta")
        scale = 0.5  # Since every logical X has two stabilizers
        for i in range(self.N_qubits - 1):
            # Hard coded XY mixer
            current_gate = XXPlusYYGate(scale * Beta)
            self.mixer_circuit.append(current_gate, [i, i + 1])

        usebarrier = self.params.get("usebarrier", False)
        if usebarrier:
            self.mixer_circuit.barrier()

    def computeFeasibleSubspace(self):
        print("Its now computing the feasible subspace")
        for combination in itertools.combinations(range(self.N_qubits), self.budget):
            current_state = ["0"] * self.N_qubits
            for index in combination:
                current_state[index] = "1"
            self.B.append("".join(current_state))

    def __XYMixerTerms(self):
        logical_X_operators = [None] * (self.N_qubits - 1)
        mixer_terms = {}
        scale = 0.5  # 1/size, size of stabilizer space
        for i in range(self.N_qubits - 2):
            logical_X_operator = ["I"] * (self.N_qubits - 1)
            logical_X_operator[i] = "X"
            logical_X_operator[i + 1] = "X"
            logical_X_operator = "".join(logical_X_operator)
            logical_X_operators[i] = logical_X_operator

            mixer_terms[logical_X_operator] = [PauliString(scale, logical_X_operator)]

            YY_operator = ["I"] * (self.N_qubits - 1)
            YY_operator[i] = "Y"
            YY_operator[i + 1] = "Y"
            YY_operator = "".join(YY_operator)

            mixer_terms[logical_X_operator].append(PauliString(scale, YY_operator))

        return mixer_terms, logical_X_operators
