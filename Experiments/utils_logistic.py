import random
import numpy as np
from logistic import f, grad_f, grad_f_i

def GD(A, y, lambd, n, d, gamma, eps = 1e-12, max_iters_num = 100000):
    x0 = np.ones(d)
    grad_norm = []
    f_vals = []
    epochs = []
    epoch = 0
    epochs.append(epoch)
    f_vals.append(f(x0, A, y, lambd, n))
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))
    while (grad_norm[-1] > eps and epochs[-1]<max_iters_num):
        x0 -= gamma*grad_f(x0, A, y, lambd, n)
        epoch += 1
        epochs.append(epoch)
        f_vals.append(f(x0, A, y, lambd, n))
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))

    return x0, f_vals, grad_norm, epochs


def GD_with_error(A, y, lambd, n, d, gamma, x_star, max_num_iter = 100):
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2    
    grad_norm = []
    epochs = []
    arg_error = []    
    epoch = 0
    epochs.append(epoch)
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)    
    for epoch in range(max_num_iter):
        x0 -= gamma*grad_f(x0, A, y, lambd, n)
        epochs.append(epoch+1)
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))
        arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    return arg_error, grad_norm, epochs


def RR_SVRG(A, y, lambd, n, d, gamma, x_star, max_num_iter = 100):
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2    
    y0 = np.ones(d)
    arg_error = []
    epochs = []
    grad_norm = []
    grad_computations = []    
    epoch = 0
    grad_comp_num = 0
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))    
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    epochs.append(epoch)
    grad_computations.append(grad_comp_num)

    for epoch in range(max_num_iter):
        grad_y = grad_f(y0, A, y, lambd, n)
        grad_comp_num += 1
        inds = random.sample(range(n), n)
        grad = 0
        grad_y_i = 0
        for ind in inds:
            grad = grad_f_i(x0, A, y, lambd, n, ind)
            grad_y_i = grad_f_i(y0, A, y, lambd, n, ind)
            x0 -= gamma * (grad - grad_y_i + grad_y)
        grad_comp_num += 1
        y0 = x0.copy()
        arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))        
        epochs.append(epoch+1)       
        grad_computations.append(grad_comp_num)
    return arg_error, grad_norm, epochs, grad_computations    


def RR_VR(A, y, lambd, n, d, gamma, x_star, p = 1, max_num_iter = 100):
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2    
    y0 = np.ones(d)
    arg_error = []
    epochs = []
    grad_norm = []
    grad_computations = []    
    epoch = 0
    grad_comp_num = 0
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))    
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    epochs.append(epoch)
    grad_computations.append(grad_comp_num)
    grad_comp_num += 1
    for epoch in range(max_num_iter):
        grad_y = grad_f(y0, A, y, lambd, n)
        inds = random.sample(range(n), n)
        grad = 0
        grad_y_i = 0
        for ind in inds:
            grad = grad_f_i(x0, A, y, lambd, n, ind)
            grad_y_i = grad_f_i(y0, A, y, lambd, n, ind)
            x0 -= gamma * (grad - grad_y_i + grad_y)
        grad_comp_num += 1
        draw = random.random()
        if (draw <= p):
            y0 = x0.copy()
            grad_comp_num += 1
            
        arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))        
        epochs.append(epoch+1)       
        grad_computations.append(grad_comp_num)
    return arg_error, grad_norm, epochs, grad_computations



def L_SVRG(A, y, lambd, n, d, gamma, x_star, p = 1, default_p = True, max_num_iter = 100):
    if (default_p):
        p = 1/n
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2    
    y0 = np.ones(d)
    arg_error = []
    epochs = []
    grad_norm = []
    grad_computations = []    
    epoch = 0
    grad_comp_num = 0
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))    
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    epochs.append(epoch)
    grad_computations.append(grad_comp_num)
    
    
    grad_y = grad_f(y0, A, y, lambd, n)
    grad_comp_num += 1
    
    for epoch in range(n*max_num_iter):
        ind = random.sample(range(n), 1)[0]
        grad = 0
        grad_y_i = 0
        x_copy = x0.copy()
        grad = grad_f_i(x0, A, y, lambd, n, ind)
        grad_y_i = grad_f_i(y0, A, y, lambd, n, ind)
        x0 -= gamma * (grad - grad_y_i + grad_y)
        grad_comp_num += 1/n
        
        if ((epoch+1)%n == 0):
            arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
            grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))        
            epochs.append((epoch+1)/n)       
            grad_computations.append(grad_comp_num)
        
        draw = random.random()
        if (draw <= p):
            y0 = x_copy.copy()
            grad_y = grad_f(y0, A, y, lambd, n)            
            grad_comp_num += 1                    
        else:
            pass        
        
    return arg_error, grad_norm, epochs, grad_computations



def SVRG(A, y, lambd, n, d, gamma, x_star, max_num_iter = 100):
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2          
    y0 = np.ones(d)
    arg_error = []
    epochs = []
    grad_norm = []
    grad_computations = []    
    epoch = 0
    grad_comp_num = 0
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))    
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    epochs.append(epoch)
    grad_computations.append(grad_comp_num)

    for epoch in range(max_num_iter):
        grad_y = grad_f(y0, A, y, lambd, n)
        grad_comp_num += 1
        grad = 0
        grad_y_i = 0
        for k in range(n):
            ind = random.sample(range(n), 1)[0]            
            grad = grad_f_i(x0, A, y, lambd, n, ind)
            grad_y_i = grad_f_i(y0, A, y, lambd, n, ind)
            x0 -= gamma * (grad - grad_y_i + grad_y)
        grad_comp_num += 1
        y0 = x0.copy()
        arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))        
        epochs.append(epoch+1)       
        grad_computations.append(grad_comp_num)
    return arg_error, grad_norm, epochs, grad_computations


def RR_SGD(A, y, lambd, n, d, gamma, x_star, max_num_iter = 100):
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2      
    grad_norm = []
    epochs = []
    arg_error = []    
    epoch = 0
    epochs.append(epoch)
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)    
    for epoch in range(max_num_iter):
        inds = random.sample(range(n), n)
        for ind in inds:
            x0 -= gamma*grad_f_i(x0, A, y, lambd, n, ind)
        epochs.append(epoch+1)
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))
        arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    return arg_error, grad_norm, epochs, epochs        

def SAGA_RR(A, y, lambd, n, d, gamma, x_star, max_num_iter = 100):
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2    
    grad_norm = []
    epochs = []
    arg_error = []
    epoch = 0
    epochs.append(epoch)
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))    
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)    
    grad_table = np.zeros((n, d))
    for epoch in range(max_num_iter):
        inds = random.sample(range(n), n) 
        for ind in inds:
            grad_ave = np.mean(grad_table, axis = 0)
            grad_table_i = grad_table[ind, :].copy()
            grad_i = grad_f_i(x0, A, y, lambd, n, ind)
            grad_table[ind, :] = grad_i.copy()
            x0 -= gamma*(grad_i - grad_table_i + grad_ave)
        epochs.append(epoch+1)
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))
        arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    return arg_error, grad_norm, epochs, epochs


def SAGA(A, y, lambd, n, d, gamma, x_star, max_num_iter = 100):
    x0 = np.ones(d)
    error0 = np.linalg.norm(x0-x_star)**2
    grad_norm = []
    epochs = []
    arg_error = []
    epoch = 0
    epochs.append(epoch)
    grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))    
    arg_error.append(np.linalg.norm(x0-x_star)**2/error0)    
    grad_table = np.zeros((n, d))
    for epoch in range(max_num_iter):
        for i in range(n):
            ind = random.sample(range(n), 1)[0]             
            grad_ave = np.mean(grad_table, axis = 0)
            grad_table_i = grad_table[ind, :].copy()
            grad_i = grad_f_i(x0, A, y, lambd, n, ind)
            grad_table[ind, :] = grad_i.copy()
            x0 -= gamma*(grad_i - grad_table_i + grad_ave)
        epochs.append(epoch+1)
        grad_norm.append(np.linalg.norm(grad_f(x0, A, y, lambd, n)))
        arg_error.append(np.linalg.norm(x0-x_star)**2/error0)
    return arg_error, grad_norm, epochs, epochs