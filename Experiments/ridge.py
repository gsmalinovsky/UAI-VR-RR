import random
import numpy as np

def f(x, A, y, lambd, n):
    return (np.linalg.norm(A@x-y)**2/(2*n) + np.linalg.norm(x)**2*lambd/2)
def grad_f(x, A, y, lambd, n):
    return (np.transpose(A)@A@x-np.transpose(A)@y)/n+lambd*x
def grad_f_i(x, A, y, lambd, n, i):
    return np.transpose(A[i,:])*(A[i,:]@x)-np.transpose(A[i,:])*y[i]+lambd*x
def optimal_rigde(A, y, lambd, n, d):
    return np.linalg.inv(np.transpose(A)@A + n*lambd*np.identity(d))@(np.transpose(A)@y)