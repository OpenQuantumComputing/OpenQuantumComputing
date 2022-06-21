import numpy as np
from sympy.physics.quantum import TensorProduct
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
from binsymbols import *
from sympy import *
import itertools
import math

X=Pauli(1)
Y=Pauli(2)
Z=Pauli(3)
I=1

def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def get_T(n, mode, d=1,i=None,j=None, shift=1, oddeven="both"):
    if mode=="leftright":
        T=np.zeros((n,n))
        if isinstance(i,list):
            if not len(i)==len(j):
                raise ValueError('mode "'+mode+'": lenght of index lists must be equal')
            for ind in range(len(i)):
                T[i[ind],j[ind]]=1
                T[j[ind],i[ind]]=1
        else:
            T[i,j]=1
            T[j,i]=1
    elif mode=="full":
        T=np.ones((n,n))
        for i in range(n):
            T[i,i]=0
    elif mode=="nearest_int":
        T=np.zeros((n,n))
        for i in range(n-shift):
            if oddeven=="even" and i%2==0:
                continue
            elif oddeven=="odd" and i%2==1:
                continue
            T[i,i+shift]=1
            T[i+shift,i]=1
    elif mode=="nearest_int_cyclic":
        T=np.zeros((n,n))
        for i in range(n-1):
            if oddeven=="even" and i%2==0:
                continue
            elif oddeven=="odd" and i%2==1:
                continue
            T[i,i+1]=1
            T[i+1,i]=1
        if not oddeven=="even":
            T[0,-1]=1
            T[-1,0]=1
    elif mode=="random":
        T=np.random.rand(n,n)
        for i in range(0,n):
            for j in range(i,n):
                T[i,j]=T[j,i]
    elif mode=="standard" or mode=="Hamming":
        ispowertwo=(n & (n-1) == 0) and n !=0
        if not ispowertwo:
            raise ValueError('mode "'+mode+'" needs n to be a power of two')
        T=np.zeros((n,n))
        log2n=int(np.log2(n))
        for i in range(n):
            for j in range(n):
                s1="{0:b}".format(i).zfill(log2n)
                s2="{0:b}".format(j).zfill(log2n)
                if hamming(s1,s2)==d:
                    T[i,j]=1
    else:
        raise NotImplementedError('mode "'+mode+'" not implemented')

    return T

class PauliStringTP:
    def __init__(self, excludeI=False):
        self.excludeI=excludeI
        self.items=[]
    def get_items_PS(self,tp):
        if isinstance(tp,Pauli):
            if not self.excludeI:
                self.items.append(tp)
            elif not tp==I:
                self.items.append(tp)
        else:
            tpL,tpR=tp.args
            if isinstance(tpL, TensorProduct):
                self.get_items_PS(tpL)
            else:
                if not self.excludeI:
                    self.items.append(tpL)
                elif not tpL==I:
                    self.items.append(tpL)
            if isinstance(tpR, TensorProduct):
                self.get_items_PS(tpR)
            else:
                if not self.excludeI:
                    self.items.append(tpR)
                elif not tpR==I:
                    self.items.append(tpR)

def HtoString(H, symbolic=False):
    ret=''
    for item in H.args:### go through all items of the sum (Pauli strings)
        if isinstance(item, Mul):### remove float
            if symbolic:
                fval,_,item = item.args
            else:
                if len(item.args)>2:
                    fval,tmp,item = item.args
                    if not math.isclose(fval,0,abs_tol=1e-7):
                        raise AssertionError("Encountered imaginary part that is not close to zero, aborting!", fval, tmp, item)
                else:
                    fval,item = item.args
                    if math.isclose(fval,0,abs_tol=1e-7):
                        item=None
                        print("depug: close to zero", fval, item)
            ret+=f'{fval:+.2f}'+" "
        if isinstance(item, TensorProduct) or isinstance(item, Pauli):### go through Pauli string
            tps=PauliStringTP()
            tps.get_items_PS(item)
            for p in tps.items:
                if p==1:
                    ret+="I"
                if p==X:
                    ret+="X"
                if p==Y:
                    ret+="Y"
                if p==Z:
                    ret+="Z"
        ret+=" "
    return ret

def num_Cnot(H, symbolic=False):
    sqg=0
    cnot=0
    for item in H.args:### go through all items of the sum (Pauli strings)
#         print(type(item))
        if isinstance(item, Mul):### remove float
            if symbolic:
                fval,_,item = item.args
            else:
                if len(item.args)>2:
                    fval,tmp,item = item.args
                    if not math.isclose(fval,0,abs_tol=1e-7):
                        raise AssertionError("Encountered imaginary part that is not close to zero, aborting!", fval, tmp, item)
                else:
                    fval,item = item.args
                    if math.isclose(fval,0,abs_tol=1e-7):
                        item=None
                        print("depug: close to zero", fval, item)
        if isinstance(item, TensorProduct) or isinstance(item, Pauli):### go through Pauli string
            tps=PauliStringTP(excludeI=True)
            tps.get_items_PS(item)
            tmp=len(tps.items)
            if tmp==1:
                sqg+=1
            elif tmp>1:
                cnot+=2*(tmp-1)
    return sqg,cnot

def get_g(binstrings):
    n=len(binstrings[0])
    x=binsymbols('x:'+str(n))
    expr=1
    for bs in binstrings:
        tmp_expr=0
        for i in range(n):
            if bs[i]=='0':
                tmp_expr+=x[i]
            else:
                tmp_expr+=(x[i]-1)**2
        expr*=tmp_expr
    return x, expand(expr)


def convert_to_ps(bs1, bs2):
    n=len(bs1)

    for j in range(n):
        if bs1[j]=="1" and bs2[j]=="0":
            tmp=1/2*(X-1j*Y)
        elif bs1[j]=="0" and bs2[j]=="1":
            tmp=1/2*(X+1j*Y)
        elif bs1[j]=="1" and bs2[j]=="1":
            tmp=1/2*(I-Z)
        else:# bs[j]=="0" and bs[j]=="0":
            tmp=1/2*(I+Z)
        if j==0:
            pauli_str=tmp
        else:
            pauli_str=TensorProduct(pauli_str,tmp)
    return pauli_str

def get_overlap(binstringsA, binstringsB):
    overlap=[]
    mA=len(binstringsA)
    mB=len(binstringsB)
    for i in range(mA):
        for j in range(mB):
            if binstringsA[i] == binstringsB[j]:
                overlap.append(binstringsA[i])
    return overlap

def get_Pauli_string(binstrings, T, symbolic=False):
    m=len(binstrings)
   
    pauli_str=0
    if symbolic:
        for i in range(m):
            for j in range(m):
                tmp_ps = convert_to_ps(binstrings[i], binstrings[j])
                pauli_str+=T[i,j]*tmp_ps
    else:
        for i in range(m):
            for j in range(m):
                if not math.isclose(T[i,j],0,abs_tol=1e-7):
                    tmp_ps = convert_to_ps(binstrings[i], binstrings[j])
                    pauli_str+=T[i,j]*tmp_ps

    return pauli_str

def simplifyH(H):
    for i in range(10):
        H = H.expand(tensorproduct=True)
    H=evaluate_pauli_product(H)
    return H

def get_H(stringlist,T,simplify=True, symbolic=False, verbose=False):
    H=get_Pauli_string(stringlist, T, symbolic=symbolic)
    if simplify:
        H=simplifyH(H)
    if verbose:
        print("#sqg, #cnots=",num_Cnot(H, symbolic=symbolic))
    return H

