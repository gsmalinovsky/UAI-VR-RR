import random
import numpy as np

def f(x, A, y, lambd, n):
    return (sum(np.log(1+np.exp((A@x)*(-y))))/n + np.linalg.norm(x)**2*lambd/2)
    
def grad_f(x, A, y, lambd, n):
    grad = 0
    for i in range(n):
        grad += -y[i]*np.exp(-y[i]*(A[i,:]@x))/(1+np.exp(-y[i]*(A[i,:]@x)))*np.transpose(A[i,:])
    grad /= n
    grad += lambd*x
    return grad
def grad_f_i(x, A, y, lambd, n, i):
    return -y[i]*np.exp(-y[i]*(A[i,:]@x))/(1+np.exp(-y[i]*(A[i,:]@x)))*np.transpose(A[i,:])+lambd*x
    
        