import numpy as np
import os
import datetime
import time
import pickle
from qiskit import *
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus

def start_or_retrieve_job(filename, backend, circuit=None, options=None):
    """function that
       1) retrieves the job from the backend if saved to file,
       2) or executes a job on a backend and saves it to file

    Parameters
    ----------
    filename : string
        The filename to write/read from. The extension ".job" is
        automatically appended to the string.

    backend : qiskit.providers.ibmq.ibmqbackend.IBMQBackend
        The backend where the job has been/is to be executed.

    circuit : qiskit.circuit.quantumcircuit.QuantumCircuit, optional
        The circuit that is to be executed.

    options: dict, optional
        The following is a list of all options and their default value
        options={'shots': 1024, 'forcererun': False, 'useapitoken': False, 'directory': 'jobs'}
        the directory is created if it does not exist

    Returns
    -------
    job : qiskit.providers.ibmq.job.ibmqjob.IBMQJob,
          qiskit.providers.aer.aerjob.AerJob
    """
    ### options parsing
    if options == None:
        options={}
    shots = options.get('shots', 1024)
    forcererun = options.get('forcererun', False)
    useapitoken = options.get('useapitoken', False)
    directory = options.get('directory', 'jobs')

    filename = filename+'.job'
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("job name:"+directory+'/'+filename)

    if not(forcererun) and os.path.isfile(directory+'/'+filename):
        #read job id from file and retrieve the job
        with open(directory+'/'+filename, 'r') as f:
            apitoken = f.readline().rstrip()
            backendname = f.readline().rstrip()
            job_id = f.readline().rstrip()
        if useapitoken:
            IBMQ.save_account(apitoken, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            backend_tmp = provider.get_backend(backendname)
            if backend.name() != backend_tmp.name():
                raise Exception("The backend of the job was "+backend_tmp.name()+", but you requested "+backend.name())
            job = backend_tmp.retrieve_job(job_id)
        else:
            job = backend.retrieve_job(job_id)
    else:
        # otherwise start the job and write the id to file
        hasnotrun = True
        while hasnotrun:
            error = False
            try:
                job = execute(circuit, backend, shots=int(shots))
            except Exception as e:
                error = True
                sec  = 60
                if "Error code: 3458" in str(e):
                    print(filename +' No credits available, retry in '+str(sec)+' seconds'+', time='+str(datetime.datetime.now()), end='\r')
                else:
                    print('{j} Error! Code: {c}, Message: {m}, Time {t}'.format(j=str(filename), c = type(e).__name__, m = str(e), t=str(datetime.datetime.now())), ", retry in ",str(sec),' seconds', end='\r')
                time.sleep(sec)
            if not(error):
                hasnotrun = False
        job_id = job.job_id()
        apitoken = IBMQ.active_account()['token']
        backendname = backend.name()
        if job_id != '':
            file = open(directory+'/'+filename,'w')
            file.write(apitoken+'\n')
            file.write(backendname+'\n')
            file.write(job_id)
            file.close()
    return job

def write_results(filename, job, options=None):
    """function that writes the results of a job to file

    Parameters
    ----------
    filename : string
        The filename to write to. The extension ".result" is
        automatically appended to the string.

    job : qiskit.providers.ibmq.job.ibmqjob.IBMQJob,
          qiskit.providers.aer.aerjob.AerJob
        The job to get the results from

    options: dict, optional
        The following is a list of all options and their default value
        options={'overwrite': False, 'directory': 'results'}

    Returns
    -------
    success : bool
        set to True if the results from the job are written to file
        it is set to False, e.g., if the job has not yet finished successfully

    """
    ### options parsing
    if options == None:
        options={}
    overwrite = options.get('overwrite', False)
    directory = options.get('directory', 'results')

    filename=filename+'.result'
    if not os.path.exists(directory):
        os.makedirs(directory)

    success = False
    fileexists = os.path.isfile(directory+'/'+filename)
    if (fileexists and overwrite) or not(fileexists):
        jobstatus = job.status()
        if jobstatus == JobStatus.DONE:
            res=job.result().results
            tmpfile = open(directory+'/'+filename,'wb')
            pickle.dump(res,tmpfile)
            tmpfile.close()
            success = True
    return success

def read_results(filename, options=None):
    """function that reads results from file

    Parameters
    ----------
    filename : string
        The filename to read from. The extension ".result" is
        automatically appended to the string.

    options: dict, optional
        The following is a list of all options and their default value
        options={'directory': 'results'}

    Returns
    -------
    results : Object
        the form is dictated by job.result().results
        can be None, if the file does not exist

    success : bool
        set to True if the results

    """
    ### options parsing
    if options == None:
        options={}
    directory = options.get('directory', 'results')

    filename=filename+'.result'

    results = None
    if os.path.isfile(directory+'/'+filename):
        tmpfile = open(directory+'/'+filename,'rb')
        results=pickle.load(tmpfile)
        tmpfile.close()
    return results

def get_id_error_rate(backend):
    errorrate=[]
    gates=backend.properties().gates
    for i in range(0,len(gates)):
        if getattr(gates[i],'gate') == 'id':
            gerror = getattr(getattr(gates[i],'parameters')[0], 'value')
            errorrate.append(gerror)
    return errorrate

def get_U3_error_rate(backend):
    errorrate=[]
    gates=backend.properties().gates
    for i in range(0,len(gates)):
        if getattr(gates[i],'gate') == 'u3':
            gerror = getattr(getattr(gates[i],'parameters')[0], 'value')
            errorrate.append(gerror)
    return errorrate

def get_T1(backend):
    val=[]
    unit=[]
    gates=backend.properties().gates
    for i in range(backend.configuration().n_qubits):
        qubit=backend.properties().qubits[i][0]
        assert qubit.name == 'T1'
        val.append(qubit.value)
        unit.append(qubit.unit)
    return val, unit

def get_T2(backend):
    val=[]
    unit=[]
    gates=backend.properties().gates
    for i in range(backend.configuration().n_qubits):
        qubit=backend.properties().qubits[i][1]
        assert qubit.name == 'T2'
        val.append(qubit.value)
        unit.append(qubit.unit)
    return val, unit

def get_readouterrors(backend):
    val=[]
    gates=backend.properties().gates
    for i in range(backend.configuration().n_qubits):
        qubit=backend.properties().qubits[i][3]
        assert qubit.name == 'readout_error'
        val.append(qubit.value)
    return val

def get_prob_meas0_prep1(backend):
    val=[]
    gates=backend.properties().gates
    for i in range(backend.configuration().n_qubits):
        qubit=backend.properties().qubits[i][4]
        assert qubit.name == 'prob_meas0_prep1'
        val.append(qubit.value)
    return val

def get_prob_meas1_prep0(backend):
    val=[]
    gates=backend.properties().gates
    for i in range(backend.configuration().n_qubits):
        qubit=backend.properties().qubits[i][5]
        assert qubit.name == 'prob_meas1_prep0'
        val.append(qubit.value)
    return val

def get_cx_error_map(backend):
    """
    function that returns a 2d array containing CX error rates.
    """
    num_qubits=backend.configuration().n_qubits
    two_qubit_error_map = np.zeros((num_qubits,num_qubits))
    backendproperties=backend.properties()
    gates=backendproperties.gates
    for i in range(0,len(gates)):
        if getattr(gates[i],'gate') == 'cx':
            cxname = getattr(gates[i],'name')
            error = getattr(getattr(gates[i],'parameters')[0], 'value')
            #print(cxname, error)
            for p in range(num_qubits):
                for q in range(num_qubits):
                    if p==q:
                        continue
                    if cxname == 'cx'+str(p)+'_'+str(q):
                        two_qubit_error_map[p][q] = error
                        break
    return two_qubit_error_map

def getNumberOfControlledGates(circuit):
    """function that returns the number of CX, CY, CZ gates.
       N.B.: swap gates are counted as 3 CX gates.
    """
    numCx=0
    numCy=0
    numCz=0
    for instr, qargs, cargs in circuit.data:
        gate_string = instr.qasm()
        if gate_string == "swap":
            numCx += 3
        elif gate_string == "cx":
            numCx += 1
        elif gate_string == "cy":
            numCy += 1
        elif gate_string == "cz":
            numCz += 1
    return numCx, numCy, numCz

def convert_to_binarystring(results):
    list=[]
    for item in range(0,len(results)):
        dict={}
        co = results[item].data.counts
        for i in range(0,2**5):
            if(hasattr(co,hex(i))):
                binstring="{0:b}".format(i).zfill(5)
                counts = getattr(co, hex(i))
                dict[binstring] = counts
        list.append(dict)
    return list


def Cn_U3_0theta0(qc, control_indices, target_index, theta):
    """
    Ref: https://arxiv.org/abs/0708.3274

    """
    n=len(control_indices)
    if n == 0:
        qc.rz(theta, control_indices)
    elif n == 1:
        qc.cu3(0, theta, 0, control_indices, target_index)
    elif n == 2:
        qc.cu3(0, theta/ 2, 0, control_indices[1], target_index)  # V gate, V^2 = U
        qc.cx(control_indices[0], control_indices[1])
        qc.cu3(0, -theta/ 2, 0, control_indices[1], target_index)  # V dagger gate
        qc.cx(control_indices[0], control_indices[1])
        qc.cu3(0, theta/ 2, 0, control_indices[0], target_index) #V gate
    else:
        raise Exception("C^nU_3(0,theta,0) not yet implemented for n="+str(n)+".")

def CGp(qc, control_index, target_index, p):
    """
    Ref: https://onlinelibrary.wiley.com/doi/pdf/10.1002/qute.201900015

    """
    thetadash = np.arcsin(np.sqrt(p))
    qc.u(thetadash, 0, 0, target_index)
    qc.cx(control_index, target_index)
    qc.u(-thetadash, 0, 0, target_index)

def Wn(qc, indices):
    """
    Ref: https://onlinelibrary.wiley.com/doi/pdf/10.1002/qute.201900015

    """
    n=len(indices)
    if n<2 or n>8:
        raise Exception("Wn not defined for n="+str(n)+".")

    qc.x(indices[0])

    if n==2:
        qc.h(indices[1])
        qc.cx(indices[1], indices[0])
    elif n==3:
        CGp(qc, indices[0], indices[1], 1/3)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[1], indices[2], 1/2)
        qc.cx(indices[2], indices[1])
    elif n==4:
        CGp(qc, indices[0], indices[1], 1/4)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[1], indices[2], 1/3)
        qc.cx(indices[2], indices[1])
        #
        CGp(qc, indices[2], indices[3], 1/2)
        qc.cx(indices[3], indices[2])
    elif n==5:
        CGp(qc, indices[0], indices[1], 2/5)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/2)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 1/3)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[3], indices[4], 1/2)
        qc.cx(indices[4], indices[3])
    elif n==6:
        CGp(qc, indices[0], indices[1], 3/6)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/3)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 2/3)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[2], indices[4], 1/2)
        qc.cx(indices[4], indices[2])
        #
        CGp(qc, indices[1], indices[5], 1/2)
        qc.cx(indices[5], indices[1])
    elif n==7:
        CGp(qc, indices[0], indices[1], 3/7)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/3)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 1/2)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[2], indices[4], 1/2)
        qc.cx(indices[4], indices[2])
        #
        CGp(qc, indices[1], indices[5], 1/2)
        qc.cx(indices[5], indices[1])
        #
        CGp(qc, indices[3], indices[6], 1/2)
        qc.cx(indices[6], indices[3])
    elif n==8:
        CGp(qc, indices[0], indices[1], 1/2)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/2)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 1/2)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[0], indices[4], 1/2)
        qc.cx(indices[4], indices[0])
        #
        CGp(qc, indices[2], indices[5], 1/2)
        qc.cx(indices[5], indices[2])
        #
        CGp(qc, indices[1], indices[6], 1/2)
        qc.cx(indices[6], indices[1])
        #
        CGp(qc, indices[3], indices[7], 1/2)
        qc.cx(indices[7], indices[3])

def HilbertSchmidtInnerProduct(M1, M2):
    return (np.dot(M1.conjugate().transpose(), M2)).trace()

PI = np.array([[1, 0],  [ 0, 1]])
PX = np.array([[0, 1],  [ 1, 0]])
PY = np.array([[0, -1j],[1j, 0]])
PZ = np.array([[1, 0],  [0, -1]])
S = [PI, PX, PY, PZ]
labels = ['I', 'X', 'Y', 'Z']

def decompose1_IZ(H):
    ps=''
    for i in [0,3]:
        a = HilbertSchmidtInnerProduct(S[i], H)/2**2
        if not np.isclose(a,0):
            label = labels[i]
            ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose2_IZ(H):
    ps=''
    for i in [0,3]:
        for j in [0,3]:
            a = HilbertSchmidtInnerProduct(np.kron(S[i], S[j]), H)/2**2
            if not np.isclose(a,0):
                label = labels[i] + labels[j]
                ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose3_IZ(H):
    ps=''
    for i in [0,3]:
        for j in [0,3]:
            for k in [0,3]:
                a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],S[k])), H)/2**3
                if not np.isclose(a,0):
                    label = labels[i] + labels[j] + labels[k]
                    ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose4_IZ(H):
    ps=''
    for i in [0,3]:
        for j in [0,3]:
            for k in [0,3]:
                for l in [0,3]:
                    a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],np.kron(S[k],S[l]))), H)/2**4
                    if not np.isclose(a,0):
                        label = labels[i] + labels[j] + labels[k] + labels[l]
                        ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose5_IZ(H):
    ps=''
    for i in [0,3]:
        for j in [0,3]:
            for k in [0,3]:
                for l in [0,3]:
                    for m in [0,3]:
                        a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],np.kron(S[k],np.kron(S[l],S[m])) )), H)
                        #/2**6
                        if not np.isclose(a,0):
                            label = labels[i] + labels[j] + labels[k] + labels[l] + labels[m]
                            ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose6_IZ(H):
    ps=''
    for i in [0,3]:
        for j in [0,3]:
            for k in [0,3]:
                for l in [0,3]:
                    for m in [0,3]:
                        for n in [0,3]:
                            a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],np.kron(S[k],np.kron(S[l],np.kron(S[m],S[n]))) )), H)
                            #/2**6
                            if not np.isclose(a,0):
                                label = labels[i] + labels[j] + labels[k] + labels[l] + labels[m] + labels[n]
                                ps+=" + "+str(a)+"*"+str(label)
    return ps


def decompose1(H):
    ps=''
    for i in range(4):
        a = HilbertSchmidtInnerProduct(S[i], H)/2**2
        if not np.isclose(a,0):
            label = labels[i]
            ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose2(H):
    ps=''
    for i in range(4):
        for j in range(4):
            a = HilbertSchmidtInnerProduct(np.kron(S[i], S[j]), H)/2**2
            if not np.isclose(a,0):
                label = labels[i] + labels[j]
                ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose3(H):
    ps=''
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],S[k])), H)/2**3
                if not np.isclose(a,0):
                    label = labels[i] + labels[j] + labels[k]
                    ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose4(H):
    ps=''
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],np.kron(S[k],S[l]))), H)/2**4
                    if not np.isclose(a,0):
                        label = labels[i] + labels[j] + labels[k] + labels[l]
                        ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose5(H):
    ps=''
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for m in range(4):
                        a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],np.kron(S[k],np.kron(S[l],S[m])) )), H)
                        #/2**6
                        if not np.isclose(a,0):
                            label = labels[i] + labels[j] + labels[k] + labels[l] + labels[m]
                            ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose6(H):
    ps=''
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for m in range(4):
                        for n in range(4):
                            a = HilbertSchmidtInnerProduct(np.kron(S[i], np.kron(S[j],np.kron(S[k],np.kron(S[l],np.kron(S[m],S[n]))) )), H)
                            #/2**6
                            if not np.isclose(a,0):
                                label = labels[i] + labels[j] + labels[k] + labels[l] + labels[m] + labels[n]
                                ps+=" + "+str(a)+"*"+str(label)
    return ps

def decompose(H, onlyIZ=False):
    nq=np.log2(H.shape[0])
    print("dim=",nq)
    if nq==1:
        if onlyIZ:
            ps=decompose1_IZ(H)
        else:
            ps=decompose1(H)
    elif nq==2:
        if onlyIZ:
            ps=decompose2_IZ(H)
        else:
            ps=decompose2(H)
    elif nq==3:
        if onlyIZ:
            ps=decompose3_IZ(H)
        else:
            ps=decompose3(H)
    elif nq==4:
        if onlyIZ:
            ps=decompose4_IZ(H)
        else:
            ps=decompose4(H)
    elif nq==5:
        if onlyIZ:
            ps=decompose5_IZ(H)
        else:
            ps=decompose5(H)
    elif nq==6:
        if onlyIZ:
            ps=decompose6_IZ(H)
        else:
            ps=decompose6(H)
    else:
        raise Exception("The method is not implemented for dim>6.")
    return ps

def get_depth_and_numCX(circuit):
    
    depth  = circuit.depth()
    num_cx = circuit.count_ops()['cx']

    return depth, num_cx
        
