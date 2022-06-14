from qiskit import *
import numpy as np

class QAOABase:

    def __init__(self):#,params = None):
        """
        init function that initializes member variables

        :param params: additional parameters
        """
        self.E=None
        self.Var=None

################################
# functions to be implemented:
################################

    def cost(self, string, params):
        """
        implements the cost function

        :param string: a binary string
        :param params: additional parameters
        :return: a scalar value
        """
        raise NotImplementedError

    def createCircuit(self, angles, depth, params={}):
        """
        implements a function to create the circuit

        :param params: additional parameters
        :return: an instance of the qiskti class QuantumCircuit
        """
        raise NotImplementedError

################################
# generic functions
################################

    def loss(self, angles, backend, depth, shots, noisemodel=None, params={}):
        """
        loss function 
        :param params: additional parameters
        :return: an instance of the qiskti class QuantumCircuit
        """
        self.opt_iterations +=1

        circuit = createCircuit(angles, depth, options=options)

        if backend.configuration().local:
            job = execute(circuit, backend=backend, noise_model=noisemodel, shots=shots)
        else:
            job = start_or_retrieve_job(name+"_"+str(opt_iterations), backend, circuit, options={'shots' : shots})

        e,v = self.measurementStatistics(job, params=params)

        opt_values[str(opt_iterations )] = e[0]
        opt_angles[str(opt_iterations )] = angles
        return -val[0]

    def measurementStatistics(self, job, params):
        """
        implements a function for expectation value and variance

        :param job: job instance derived from BaseJob
        :return: expectation and variance
        """
        expectations = []
        variances = []

        jres=job.result()
        counts_list=jres.get_counts()

        for i, counts in enumerate(counts_list):
            n_shots = jres.results[i].shots
            E = 0
            E2 = 0
            for string in counts:
                cost = self.cost(string, params)
                E += cost*counts[string]
                E2 += cost**2*counts[string];
            if n_shots == 1:
                v = 0
            else:
                E/=n_shots
                E2/=n_shots
                v = (E2-E**2)*n_shots/(n_shots-1)
            expectations.append(E)
            variances.append(v)
        return expectations, variances

    def interp(self, angles):
        """
        INTERP heuristic/linear interpolation for initial parameters
        when going from depth p to p+1 (https://doi.org/10.1103/PhysRevX.10.021067)
        E.g. [0., 2., 3., 6., 11., 0.] becomes [2., 2.75, 4.5, 7.25, 11.]


        :param angles: angles for depth p
        :return: linear interpolation of angles for depth p+1
        """
        tmp=np.zeros(len(angles)+2)
        tmp[1:-1]=angles
        w=np.arange(0,p+1)
        return w/p*tmp[:-1] + (p-w)/p*tmp[1:]

    def sample_cost_landscape(self, backend, shots, noisemodel=None, params={}, verbose=True, angles={"gamma": [0,2*np.pi,20], "beta": [0,2*np.pi,20]}):
        if verbose:
            print("Calculating Energy landscape for depth p=1...")

        depth=1

        tmp=angles["gamma"]
        gamma_grid = np.linspace(tmp[0],tmp[1],tmp[2])
        tmp=angles["beta"]
        beta_grid = np.linspace(tmp[0],tmp[1],tmp[2])

        if backend.configuration().local:
            circuits=[]
            for beta in beta_grid:
                for gamma in gamma_grid:
                    #params['name'] = str(beta_n)+"_"+str(gamma_n)
                    circuits.append(self.createCircuit(np.array((gamma,beta)), depth, params=params))
            job = execute(circuits, backend, shots=shots)
            e, v = self.measurementStatistics(job, params=params)
            self.E = -np.array(e).reshape(angles["beta"][2],angles["gamma"][2])
            self.Var = np.array(v).reshape(angles["beta"][2],angles["gamma"][2])
        else:
            self.E = np.zeros((angles["beta"][2],angles["gamma"][2]))
            b=-1
            for beta in beta_grid:
                b+=1
                g=-1
                for gamma in gamma_grid:
                    g+=1
                    params['name'] = str(b)+"_"+str(g)
                    circuit = createCircuit(np.array((gamma,beta)), depth, options=params)
                    job = start_or_retrieve_job(name+"_"+str(b)+"_"+str(g), backend, circuit, options={'shots' : shots})
                    e,_ = self.measurementStatistics(job, params=params)
                    self.E[b,g] = -e[0]
        if verbose:
            print("Calculating Energy landscape done")

        return self.E

    def local_opt(self, backend, depth, shots, noisemodel=None, options={}, maxdepth=3):

        self.depth = 1

        E = self.sample_cost_landscape(backend, dpeth, shots, noisemodel=noisemodel, options=options)

        while self.continue_simulation():
            # Reset the current book-keeping variables for each depth
            self.init_optimization_statistic_params()

            # Local optimisation

            for rep in range(self.repeats):
                print(f"Depth = {self.depth}, Rep = {rep + 1}")

                # No need to keep track of the optimisation result, as the getval-function
                # is required to update the member variables opt_values, opt_best_values, opt_params
                # among all iterations, so that multiple repetitions can be performed and compared.
                #
                # The function save_best_params() will ensure the best of each repetition will
                # be used further.
                _ = optimize.minimize(self.getval,
                                      x0 = x0,
                                      method = self.optmethod,
                                      options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})

            self.save_best_params()
            self.depth += 1

            # Extrapolate the parameters to the next depth
            x0 = self.interp_init()

        return Elandscape, self.params, self.E, self.best
