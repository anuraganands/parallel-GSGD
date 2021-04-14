import numpy as np
from multiprocessing import current_process
import time
from random import randint
import ray

def logistic_func(beta, X):
    ''' 
    logistic(sigmoid) function 
    '''
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))

@ray.remote
def log_gradient(p, X, y):
    ''' 
    logistic gradient function 
    '''
    
    w = ray.get(p.getweight.remote())
    first_calc = logistic_func(w, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X)
    update = w - (.2 * final_calc)
    p.updatewight.remote(update)
    


def cost_func(beta, X, y):
    ''' 
    cost function, J 
    '''
    log_func_v = logistic_func(beta, X)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)


def grad_desc(X, y, beta, lr=.01, converge_change=.001):
    ''' 
    gradient descent function 

    cost = cost_func(beta, X, y)
    change_cost = 1
    num_iter = 1
    m = X.shape[0]
    z = np.arange(m)
    while(num_iter < 100):
        old_cost = cost
        z = np.random.permutation(z)
        grad = log_gradient(beta, X[z], y[z])
        beta = beta - (lr * grad)
        cost = cost_func(beta, X, y)
        change_cost = old_cost - cost
        print(num_iter)
        num_iter += 1

    return beta, num_iter
    '''


def pred_values(beta, X):
    ''' 
    function to predict labels 
    '''
    pred_prob = logistic_func(beta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    return np.squeeze(pred_value)
