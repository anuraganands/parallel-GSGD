import csv
import numpy as np
import matplotlib.pyplot as plt
#import ray
import ray
import time
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pandas import read_csv, DataFrame
import multiprocessing as mp
import random

def loadCSV(filename):
    '''
    function to load dataset
    '''
    lencoder = preprocessing.LabelEncoder()
    dataset = read_csv("adult.csv",header=None)
    dataset = DataFrame(dataset)

    dataset.loc[:,:] = dataset.loc[:,:].replace("?",np.nan)
    dataset.fillna(dataset.mean(),inplace=True)
    dataset.dropna(inplace=True)

    x = [1,3,5,6,7,8,9,13,14]
    for i in x:
        dataset.loc[:,i] = lencoder.fit_transform(dataset.loc[:,i])
    
    print(dataset.head(10))
    dataset = np.array(dataset)
    return np.array(dataset)


def normalize(X):
    '''
    function to normalize feature matrix, X
    '''
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    norm_X = 1 - ((maxs - X)/rng)
    return norm_X


def logistic_func(beta, X):
    '''
    logistic(sigmoid) function
    '''
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))


def log_gradient(beta, X, y):
    '''
    logistic gradient function
    '''
    first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X)
    return final_calc


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


def grad_desc(X, y, beta, lr=.1, converge_change=.001):
    '''
    gradient descent function
    '''
    cost = cost_func(beta, X, y)
    change_cost = 1
    num_iter = 1
    m = X.shape[0]
    z = np.arange(m)
    while(num_iter < 2000):
        z = np.random.permutation(z)
        #old_cost = cost
        grad = log_gradient(beta, X[z[0]], y[z[0]])
        beta = beta - (lr * grad)
        #cost = cost_func(beta, X, y)
        #change_cost = old_cost - cost
        num_iter += 1

    return beta, num_iter


def pred_values(beta, X):
    ''' 
    function to predict labels 
    '''
    pred_prob = logistic_func(beta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    return np.squeeze(pred_value)


def plot_reg(X, y, beta):
    ''' 
    function to plot decision boundary 
    '''
    # labelled observations
    x_0 = X[np.where(y == 0.0)]
    x_1 = X[np.where(y == 1.0)]

    # plotting points with diff color for diff label
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')

    # plotting decision boundary
    x1 = np.arange(0, 1, 0.1)
    x2 = -(beta[0, 0] + beta[0, 1]*x1)/beta[0, 2]
    plt.plot(x1, x2, c='k', label='reg line')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # load the dataset
    dataset = loadCSV('bc.csv')
    starttime = time.time()
    # normalizing feature matrix
    X = normalize(dataset[:, :-1])

    # stacking columns wth all ones in feature matrix
    X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))

    # response vector
    y = dataset[:, -1]


    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.4, random_state=0)

    # initial beta values
    w = np.matrix(np.zeros(xtrain.shape[1]))

    # beta values after running gradient descent
    beta, num_iter = grad_desc(xtrain, ytrain, w)

    # estimated beta values and number of iterations
    print("Estimated regression coefficients:", beta)
    print("No. of iterations:", num_iter)

    # predicted labels
    y_pred = pred_values(beta, xtest)

    # number of correctly predicted labels
    print("Correctly predicted labels:", np.sum(ytest == y_pred))
    print("Accuracy", round(np.sum(ytest == y_pred)/len(ytest) * 100, 2), "%")
    print("Time Taken: ", time.time()-starttime)
    # plotting regression line
    # plot_reg(X, y, beta)
