from qiskit import *
import numpy as np
from scipy.optimize import minimize
import math

from openquantumcomputing.Statistic import Statistic

class QAOABase:

    def __init__(self,params = None):
        """
        init function that initializes member variables

        :param params: additional parameters
        """
        self.params=params
        self.E=None
        self.Var=None
        self.current_depth=0 # depth at which local optimization has been done
        self.angles_hist={} # initial and final angles during optimization per depth
        self.num_fval={} # number of function evaluations per depth
        self.num_shots={} # number of total shots taken for local optimization per depth
        self.costval={} # optimal cost values per depth
        self.gamma_grid=None
        self.beta_grid=None
        self.stat=Statistic()

        self.g_it=0
        self.g_values={}
        self.g_angles={}

################################
# functions to be implemented:
################################

    def cost(self, string):
        """
        implements the cost function

        :param string: a binary string
        :return: a scalar value
        """
        raise NotImplementedError

    def createCircuit(self, angles, depth):
        """
        implements a function to create the circuit

        :return: an instance of the qiskti class QuantumCircuit
        """
        raise NotImplementedError

################################
# generic functions
################################

    def isFeasible(self, string):
        """
        needs to be implemented to run successProbability
        """
        return True

    def successProbability(self, angles, backend, shots, noisemodel=None):
        """
        success is defined through cost function to be equal to 0
        """
        depth=int(len(angles)/2)
        circ=self.createCircuit(angles, depth)
        if backend.configuration().local:
            job = execute(circ, backend, shots=shots)
        else:
            job = start_or_retrieve_job("sprob", backend, circ, options={'shots' : shots})

        jres=job.result()
        counts_list=jres.get_counts()
        if isinstance(counts_list, list):
            s_prob=[]
            for i, counts in enumerate(counts_list):
                tmp=0
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    if self.isFeasible(string[::-1]):
                        tmp+=counts[string]
                s_prob.append(tmp)
        else:
            s_prob=0
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                if self.isFeasible(string[::-1]):
                    s_prob+=counts_list[string]
        return s_prob/shots

    def loss(self, angles, backend, depth, shots, precision, noisemodel):
        """
        loss function
        :return: an instance of the qiskti class QuantumCircuit
        """
        self.g_it+=1
        circuit = self.createCircuit(angles, depth)

        n_target=shots
        self.stat.reset()
        shots_taken=0

        for i in range(3):
            if backend.configuration().local:
                job = execute(circuit, backend=backend, noise_model=noisemodel, shots=shots)
            else:
                name=""
                job = start_or_retrieve_job(name+"_"+str(opt_iterations), backend, circuit, options={'shots' : shots})
            shots_taken+=shots
            _,_ = self.measurementStatistics(job)
            if precision is None:
                break
            else:
                v=self.stat.get_Variance()
                shots=int((np.sqrt(v)/precision)**2)-shots_taken
                if shots<=0:
                    break

        self.num_shots['d'+str(self.current_depth+1)]+=shots_taken

        self.g_values[str(self.g_it)] = -self.stat.get_E()
        self.g_angles[str(self.g_it)] = angles.copy()

        #opt_values[str(opt_iterations )] = e[0]
        #opt_angles[str(opt_iterations )] = angles
        return -self.stat.get_E()

    def measurementStatistics(self, job):
        """
        implements a function for expectation value and variance

        :param job: job instance derived from BaseJob
        :return: expectation and variance
        """
        jres=job.result()
        counts_list=jres.get_counts()
        if isinstance(counts_list, list):
            expectations = []
            variances = []
            for i, counts in enumerate(counts_list):
                self.stat.reset()
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    cost = self.cost(string[::-1])
                    self.stat.add_sample(cost, counts[string])
                expectations.append(self.stat.get_E())
                variances.append(self.stat.get_Variance())
            return expectations, variances
        else:
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                cost = self.cost(string[::-1])
                self.stat.add_sample(cost, counts_list[string])
            return self.stat.get_E(), self.stat.get_Variance()

    def hist(self, angles, backend, shots, noisemodel=None):
        depth=int(len(angles)/2)
        circ=self.createCircuit(angles, depth)
        if backend.configuration().local:
            job = execute(circ, backend, shots=shots)
        else:
            job = start_or_retrieve_job("hist", backend, circ, options={'shots' : shots})
        return job.result().get_counts()

    def random_init(self, gamma_bounds,beta_bounds,depth):
        """
        Enforces the bounds of gamma and beta based on the graph type.
        :param gamma_bounds: Parameter bound tuple (min,max) for gamma
        :param beta_bounds: Parameter bound tuple (min,max) for beta
        :return: np.array on the form (gamma_1, beta_1, gamma_2, ...., gamma_d, beta_d)
        """
        gamma_list = np.random.uniform(gamma_bounds[0],gamma_bounds[1], size=depth)
        beta_list = np.random.uniform(beta_bounds[0],beta_bounds[1], size=depth)
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
        depth=len(angles)
        tmp=np.zeros(len(angles)+2)
        tmp[1:-1]=angles.copy()
        w=np.arange(0,depth+1)
        return w/depth*tmp[:-1] + (depth-w)/depth*tmp[1:]

    def sample_cost_landscape(self, backend, shots=1024, noisemodel=None, verbose=True, angles={"gamma": [0,2*np.pi,20], "beta": [0,2*np.pi,20]}):
        if verbose:
            print("Calculating Energy landscape for depth p=1...")

        depth=1

        tmp=angles["gamma"]
        self.gamma_grid = np.linspace(tmp[0],tmp[1],tmp[2])
        tmp=angles["beta"]
        self.beta_grid = np.linspace(tmp[0],tmp[1],tmp[2])

        if backend.configuration().local:
            circuits=[]
            for beta in self.beta_grid:
                for gamma in self.gamma_grid:
                    circuits.append(self.createCircuit(np.array((gamma,beta)), depth))
            job = execute(circuits, backend, shots=shots)
            e, v = self.measurementStatistics(job)
            self.E = -np.array(e).reshape(angles["beta"][2],angles["gamma"][2])
            self.Var = np.array(v).reshape(angles["beta"][2],angles["gamma"][2])
        else:
            self.E = np.zeros((angles["beta"][2],angles["gamma"][2]))
            self.Var = np.zeros((angles["beta"][2],angles["gamma"][2]))
            b=-1
            for beta in self.beta_grid:
                b+=1
                g=-1
                for gamma in self.gamma_grid:
                    g+=1
                    circuit = createCircuit(np.array((gamma,beta)), depth)
                    name=""
                    job = start_or_retrieve_job(name+"_"+str(b)+"_"+str(g), backend, circuit, options={'shots' : shots})
                    e,v = self.measurementStatistics(job)
                    self.E[b,g] = -e[0]
                    self.Var[b,g] = -v[0]

        #self.current_depth=1
        if verbose:
            print("Calculating Energy landscape done")

    def get_current_deptgh(self):
        return self.current_depth

    def local_opt(self, angles0, backend, shots, precision, noisemodel=None, method='COBYLA'):
        """

        :param angles0: initial guess
        """

        depth=int(len(angles0)/2)

        self.num_shots['d'+str(self.current_depth+1)]=0
        res = minimize(self.loss, x0 = angles0, method = method,
                       args=(backend, depth, shots, precision, noisemodel))
        return res

    def increase_depth(self, backend, shots=1024, precision=None, noisemodel=None, method='COBYLA'):
        """
        sample cost landscape

        :param backend: backend
        :param shots: if precision=None, the number of samples taken
                      if precision!=None, the minimum number of samples taken
        :param precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        """

        if self.current_depth == 0:
            if self.E is None:
                self.sample_cost_landscape(backend, shots, noisemodel=noisemodel, angles={"gamma": [0,2*np.pi,20], "beta": [0,2*np.pi,20]})
            ind_Emin = np.unravel_index(np.argmin(self.E, axis=None), self.E.shape)
            angles0=np.array((self.gamma_grid[ind_Emin[1]], self.beta_grid[ind_Emin[0]]))
            self.angles_hist['d1_initial']=angles0
        else:
            gamma=self.angles_hist['d'+str(self.current_depth)+'_final'][::2]
            beta=self.angles_hist['d'+str(self.current_depth)+'_final'][1::2]
            gamma_interp=self.interp(gamma)
            beta_interp=self.interp(beta)
            angles0=np.zeros(2*(self.current_depth+1))
            angles0[::2]=gamma_interp
            angles0[1::2]=beta_interp
            self.angles_hist['d'+str(self.current_depth+1)+'_initial']=angles0

        self.g_it=0
        self.g_values={}
        self.g_angles={}

        res = self.local_opt(angles0, backend, shots, precision, noisemodel=noisemodel, method=method)
        if not res.success:
            raise Warning("Local optimization was not successful.", res)
        self.num_fval['d'+str(self.current_depth+1)]=res.nfev
        print("cost(depth=",self.current_depth+1,")=", res.fun)

        ind = min(self.g_values, key=self.g_values.get)
        self.angles_hist['d'+str(self.current_depth+1)+'_final']=self.g_angles[ind]
        self.costval['d'+str(self.current_depth+1)+'_final']=self.g_values[ind]


        self.current_depth+=1

