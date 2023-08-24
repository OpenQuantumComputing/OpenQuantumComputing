from qiskit import *
import numpy as np
import math, time
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import *  ### pip install qiskit-algorithms
from qiskit.primitives import Sampler

from openquantumcomputing.Statistic import Statistic


class QAOABase:
    def __init__(self, params=None):
        """
        init function that initializes member variables

        :param params: additional parameters

        :param backend: backend
        :param shots: if precision=None, the number of samples taken
                      if precision!=None, the minimum number of samples taken
        :param precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        """
        self.params = params
        self.E = None

        self.Var = None
        self.isQNSPSA = False

        self.optimizer = self.params.get("optimizer", [COBYLA, {}])

        qasm_sim = Aer.get_backend("qasm_simulator")
        self.backend = self.params.get("backend", qasm_sim)
        self.shots = self.params.get("shots", 1024)
        self.noisemodel = self.params.get("noisemodel", None)
        self.precision = self.params.get("precision", None)

        self.current_depth = 0  # depth at which local optimization has been done
        self.angles_hist = {}  # initial and final angles during optimization per depth
        self.num_fval = {}  # number of function evaluations per depth
        self.t_per_fval = {}  # wall time per function evaluation per depth
        self.num_shots = (
            {}
        )  # number of total shots taken for local optimization per depth
        self.costval = {}  # optimal cost values per depth
        self.gamma_grid = None
        self.beta_grid = None
        self.stat = Statistic(alpha=self.params.get("alpha", 1))
        self.N_qubits = None  # Needs to be initialized in a child class

        # Related to parameterized circuit
        self.parameterized_circuit = None
        self.parametrized_circuit_depth = 0
        self.gamma_params = None
        self.beta_params = None
        self.mixer_circuit = None
        self.cost_circuit = None

        self.g_it = 0
        self.g_values = {}
        self.g_angles = {}

    ################################
    # functions to be implemented:
    ################################

    def cost(self, string):
        """
        implements the cost function

        :param string: a binary string
        :return: a scalar value
        """

    def create_cost_circuit(self):
        """
        Implements the function that initializes the member variable
        self.cost_circuit as a parameterized circuit

        """
        raise NotImplementedError

    def create_mixer_circuit(self):
        """
        Implements the function that initializes the member variable
        self.mixer_circuit as a parameterized circuit

        Overwritten in child classes where a constraint preserving mixer is used, for example the XY-mixer
        """

        q = QuantumRegister(self.N_qubits)
        mixer_param = Parameter("x_beta")

        self.mixer_circuit = QuantumCircuit(q)
        self.mixer_circuit.rx(-2 * mixer_param, range(self.N_qubits))

        usebarrier = self.params.get("usebarrier", False)
        if usebarrier:
            self.mixer_circuit.barrier()

    def setToInitialState(self, q):
        """
        Implements the function that sets the member variable self.parameterized_circuit
        in the initial state
        :param q: The qubit register which is initialized

        Is overwritten for child classes where initial state should be superposition
        over feasible states

        """
        self.parameterized_circuit.h(range(self.N_qubits))

    ################################
    # generic functions
    ################################

    def createParameterizedCircuit(self, depth):
        """
        Implements a function to create a parameterized circuit.
        Initializes the member variable parameterized_circuit

        :param depth: depth of parameterized circuit
        """

        if self.cost_circuit == None:
            self.create_cost_circuit()

        if self.mixer_circuit == None:
            self.create_mixer_circuit()

        if self.parametrized_circuit_depth != depth:
            q = QuantumRegister(self.N_qubits)
            c = ClassicalRegister(self.N_qubits)
            self.parameterized_circuit = QuantumCircuit(q, c)

            self.gamma_params = [None] * depth
            self.beta_params = [None] * depth

            ### Initial state
            self.setToInitialState(q)

            for d in range(depth):
                self.gamma_params[d] = Parameter("gamma_" + str(d))
                cost_circuit = self.cost_circuit.assign_parameters(
                    {self.cost_circuit.parameters[0]: self.gamma_params[d]},
                    inplace=False,
                )
                self.parameterized_circuit.compose(cost_circuit, inplace=True)

                self.beta_params[d] = Parameter("beta_" + str(d))
                mixer_circuit = self.mixer_circuit.assign_parameters(
                    {self.mixer_circuit.parameters[0]: self.beta_params[d]},
                    inplace=False,
                )
                self.parameterized_circuit.compose(mixer_circuit, inplace=True)

            self.parameterized_circuit.barrier()
            self.parameterized_circuit.measure(q, c)
            self.parametrized_circuit_depth = depth
        else:
            # Dont need to do anything if current circuit is the correct depth
            pass

    def isFeasible(self, string):
        """
        needs to be implemented to run successProbability
        """
        return True

    def successProbability(self, angles):
        """
        success is defined through cost function to be equal to 0
        """
        depth = int(len(angles) / 2)
        self.createParameterizedCircuit(depth)
        params = self.getParametersToBind(angles, depth, asList=True)
        if self.backend.configuration().local:
            job = execute(
                self.parameterized_circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[params],
                optimization_level=0,
            )
        else:
            job = start_or_retrieve_job(
                "sprob",
                self.backend,
                self.parameterized_circuit,
                options={"shots": self.shots},
            )  # Now sending in a parameterized circuit here

        jres = job.result()
        counts_list = jres.get_counts()
        if isinstance(counts_list, list):
            s_prob = []
            for i, counts in enumerate(counts_list):
                tmp = 0
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    if self.isFeasible(string[::-1]):
                        tmp += counts[string]
                s_prob.append(tmp)
        else:
            s_prob = 0
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                if self.isFeasible(string[::-1]):
                    s_prob += counts_list[string]
        return s_prob / self.shots

    def getParametersToBind(self, angles, depth, asList=False):
        """
        Utility function to structure the parameterized parameter values
        so that they can be applied/bound to the parameterized circuit.

        :param angles: gamma and beta values
        :param depth: circuit depth
        :asList: Boolean that specify if the values in the dict should be a list or not
        :return: A dict containing parameters as keys and parameter values as values
        """
        assert len(angles) == 2 * depth

        params = {}
        for d in range(depth):
            if asList:
                params[self.gamma_params[d]] = [angles[2 * d + 0]]
                params[self.beta_params[d]] = [angles[2 * d + 1]]
            else:
                params[self.gamma_params[d]] = angles[2 * d + 0]
                params[self.beta_params[d]] = angles[2 * d + 1]
        return params

    def _applyParameters(self, angles, depth):
        """
        Wrapper for binding the given parameters to a parameterized circuit.
        Best used when evaluating a single circuit, as is the case in the optimization loop.
        """
        params = self.getParametersToBind(angles, depth)
        return self.parameterized_circuit.bind_parameters(params)

    def loss(self, angles):
        """
        loss function
        :return: an instance of the qiskit class QuantumCircuit
        """
        self.g_it += 1

        # depth = int(len(angles) / 2)

        circuit = None
        n_target = self.shots
        self.stat.reset()
        shots_taken = 0
        shots = self.shots

        for i in range(3):
            if self.backend.configuration().local:
                params = self.getParametersToBind(
                    angles, self.parametrized_circuit_depth, asList=True
                )
                job = execute(
                    self.parameterized_circuit,
                    backend=self.backend,
                    noise_model=self.noisemodel,
                    shots=shots,
                    parameter_binds=[params],
                    optimization_level=0,
                )
            else:
                name = ""
                job = start_or_retrieve_job(
                    name + "_" + str(opt_iterations),
                    self.backend,
                    circuit,
                    options={"shots": shots},
                )
            shots_taken += shots
            _, _ = self.measurementStatistics(job)
            if self.precision is None:
                break
            else:
                v = self.stat.get_Variance()
                shots = int((np.sqrt(v) / self.precision) ** 2) - shots_taken
                if shots <= 0:
                    break

        self.num_shots["d" + str(self.current_depth + 1)] += shots_taken

        self.g_values[str(self.g_it)] = -self.stat.get_CVaR()
        self.g_angles[str(self.g_it)] = angles.copy()

        # opt_values[str(opt_iterations )] = e[0]
        # opt_angles[str(opt_iterations )] = angles
        return -self.stat.get_CVaR()

    def measurementStatistics(self, job):
        """
        implements a function for expectation value and variance

        :param job: job instance derived from BaseJob
        :return: expectation and variance
        """
        jres = job.result()
        counts_list = jres.get_counts()
        if isinstance(counts_list, list):
            expectations = []
            variances = []
            for i, counts in enumerate(counts_list):
                self.stat.reset()
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    cost = self.cost(string[::-1])
                    self.stat.add_sample(cost, counts[string])
                expectations.append(self.stat.get_CVaR())
                variances.append(self.stat.get_Variance())
            return expectations, variances
        else:
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                cost = self.cost(string[::-1])
                self.stat.add_sample(cost, counts_list[string])
            return self.stat.get_CVaR(), self.stat.get_Variance()

    def hist(self, angles):
        depth = int(len(angles) / 2)
        self.createParameterizedCircuit(depth)

        params = self.getParametersToBind(angles, depth, asList=True)
        if self.backend.configuration().local:
            job = execute(
                self.parameterized_circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[params],
                optimization_level=0,
            )
        else:
            job = start_or_retrieve_job(
                "hist", self.backend, circ, options={"shots": self.shots}
            )
        return job.result().get_counts()

    def random_init(self, gamma_bounds, beta_bounds, depth):
        """
        Enforces the bounds of gamma and beta based on the graph type.
        :param gamma_bounds: Parameter bound tuple (min,max) for gamma
        :param beta_bounds: Parameter bound tuple (min,max) for beta
        :return: np.array on the form (gamma_1, beta_1, gamma_2, ...., gamma_d, beta_d)
        """
        gamma_list = np.random.uniform(gamma_bounds[0], gamma_bounds[1], size=depth)
        beta_list = np.random.uniform(beta_bounds[0], beta_bounds[1], size=depth)
        initial = np.empty((gamma_list.size + beta_list.size,), dtype=gamma_list.dtype)
        initial[0::2] = gamma_list
        initial[1::2] = beta_list
        return initial

    def interp(self, angles):
        """
        INTERP heuristic/linear interpolation for initial parameters
        when going from depth p to p+1 (https://doi.org/10.1103/PhysRevX.10.021067)
        E.g. [0., 2., 3., 6., 11., 0.] becomes [2., 2.75, 4.5, 7.25, 11.]

        :param angles: angles for depth p
        :return: linear interpolation of angles for depth p+1
        """
        depth = len(angles)
        tmp = np.zeros(len(angles) + 2)
        tmp[1:-1] = angles.copy()
        w = np.arange(0, depth + 1)
        return w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]

    def sample_cost_landscape(
        self,
        verbose=True,
        angles={"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]},
    ):
        if verbose:
            print("Calculating Energy landscape for depth p=1...")

        depth = 1

        tmp = angles["gamma"]
        self.gamma_grid = np.linspace(tmp[0], tmp[1], tmp[2])
        tmp = angles["beta"]
        self.beta_grid = np.linspace(tmp[0], tmp[1], tmp[2])

        if self.backend.configuration().local:
            self.createParameterizedCircuit(depth)
            # parameters = []
            gamma = [None] * angles["beta"][2] * angles["gamma"][2]
            beta = [None] * angles["beta"][2] * angles["gamma"][2]

            counter = 0
            for b in range(angles["beta"][2]):
                for g in range(angles["gamma"][2]):
                    gamma[counter] = self.gamma_grid[g]
                    beta[counter] = self.beta_grid[b]
                    counter += 1

            parameters = {self.gamma_params[0]: gamma, self.beta_params[0]: beta}

            print("Executing sample_cost_landscape")
            print("parameters: ", len(parameters))
            job = execute(
                self.parameterized_circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[parameters],
                optimization_level=0,
            )
            print("Done execute")
            e, v = self.measurementStatistics(job)
            print("Done measurement")
            self.E = -np.array(e).reshape(angles["beta"][2], angles["gamma"][2])
            self.Var = np.array(v).reshape(angles["beta"][2], angles["gamma"][2])

        else:
            self.E = np.zeros((angles["beta"][2], angles["gamma"][2]))
            self.Var = np.zeros((angles["beta"][2], angles["gamma"][2]))
            b = -1
            for beta in self.beta_grid:
                b += 1
                g = -1
                for gamma in self.gamma_grid:
                    g += 1
                    name = ""
                    job = start_or_retrieve_job(
                        name + "_" + str(b) + "_" + str(g),
                        self.backend,
                        self.parameterized_circuit,
                        options={"shots": self.shots},
                    )  # Now takes in parameterized_circuit
                    e, v = self.measurementStatistics(job)
                    self.E[b, g] = -e[0]
                    self.Var[b, g] = -v[0]

        # self.current_depth=1
        if verbose:
            print("Calculating Energy landscape done")

    def local_opt(self, angles0):
        """

        :param angles0: initial guess
        """

        self.num_shots["d" + str(self.current_depth + 1)] = 0

        try:
            opt = self.optimizer[0](**self.optimizer[1])
        except TypeError as e:  ### QNSPSA needs fidelity
            self.isQNSPSA=True
            self.optimizer[1]["fidelity"] = self.optimizer[0].get_fidelity(
                self.parameterized_circuit, sampler=Sampler()
            )
            opt = self.optimizer[0](**self.optimizer[1])
        res = opt.minimize(self.loss, x0=angles0)
        if self.isQNSPSA:
            self.optimizer[1].pop("fidelity")
        return res

    def increase_depth(self):
        """
        sample cost landscape
        """

        t_start = time.time()
        if self.current_depth == 0:
            if self.E is None:
                self.sample_cost_landscape(
                    angles={"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]},
                )
            ind_Emin = np.unravel_index(np.argmin(self.E, axis=None), self.E.shape)
            angles0 = np.array(
                (self.gamma_grid[ind_Emin[1]], self.beta_grid[ind_Emin[0]])
            )
            self.angles_hist["d1_initial"] = angles0
        else:
            gamma = self.angles_hist["d" + str(self.current_depth) + "_final"][::2]
            beta = self.angles_hist["d" + str(self.current_depth) + "_final"][1::2]
            gamma_interp = self.interp(gamma)
            beta_interp = self.interp(beta)
            angles0 = np.zeros(2 * (self.current_depth + 1))
            angles0[::2] = gamma_interp
            angles0[1::2] = beta_interp
            self.angles_hist["d" + str(self.current_depth + 1) + "_initial"] = angles0

        self.g_it = 0
        self.g_values = {}
        self.g_angles = {}
        # Create parameterized circuit at new depth
        self.createParameterizedCircuit(int(len(angles0) / 2))

        res = self.local_opt(angles0)
        # if not res.success:
        #    raise Warning("Local optimization was not successful.", res)
        self.num_fval["d" + str(self.current_depth + 1)] = res.nfev
        self.t_per_fval["d" + str(self.current_depth + 1)] = (
            time.time() - t_start
        ) / res.nfev
        print("cost(depth=", self.current_depth + 1, ")=", res.fun)

        ind = min(self.g_values, key=self.g_values.get)
        self.angles_hist["d" + str(self.current_depth + 1) + "_final"] = self.g_angles[
            ind
        ]
        self.costval["d" + str(self.current_depth + 1) + "_final"] = self.g_values[ind]

        self.current_depth += 1
