import numpy as np


def normalize(X):
    ''' 
    function to normalize feature matrix, X 
    '''
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    norm_X = 1 - ((maxs - X)/rng)
    return norm_X
