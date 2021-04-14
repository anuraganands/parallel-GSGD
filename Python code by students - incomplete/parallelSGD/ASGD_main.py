import numpy as np
from ASGD_read_csv import loadCSV
from ASGD_normalize import normalize
from ASGD_plot_error import plot_reg
from ASGD_logistic_func import logistic_func, log_gradient, cost_func, grad_desc, pred_values
from ParameterServer import ParameterServer
import ray
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import statistics
import matplotlib.pyplot as plt
import random


if __name__ == "__main__":
    # load the dataset
    sync = False
    fig = plt.figure()
    #files_list = ['bc.csv','cardio_train.csv', 'diabetes.csv','Liver_disorder.csv']
    files_list = ['diabetes.csv']
    for i in files_list:
        print('Loading file : ', i)
        dataset = loadCSV(i)

        # normalizing feature matrix
        X = normalize(dataset[:, :-1])
        #print(X)
        # stacking columns wth all ones in feature matrix
        X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
        
        # response vector
        y = dataset[:, -1]

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.4, random_state=0)

        # initial beta values
        #w = np.matrix(np.zeros(X.shape[1]))
        w = np.matrix(np.zeros(xtrain.shape[1]))
        acclist = []
        timelist = []

        for _ in range(1):
            starttime = time.time()
            if(sync):
                ray.init(local_mode=True)
            else:
                ray.init(num_cpus=4)
            
            p = ParameterServer.remote(xtrain, ytrain, w)


            num_iter = 1
            m = xtrain.shape[0]
            z = np.arange(m)
            worker_tasks = []
            #random.seed(1234) # seed the random generator for reproducing the results
            
            while(num_iter < 1000):
                z = np.random.permutation(z)
                #print(z[0])
                
                worker_tasks.append(log_gradient.remote(p, xtrain[z[0]], ytrain[z[0]]))
                #z = random.sample(list(z), 100)
                
                #worker_tasks.append(log_gradient.remote(p, xtrain[z], ytrain[z]))
                #random.sample(the_list, 50)
                # print(num_iter)
                num_iter += 1

            res = []
            count = 0
            for i in tqdm(worker_tasks):
                # print(i)
                ready, not_ready = ray.wait(worker_tasks)
                worker_tasks = not_ready
                #new_messages = ray.get(p.getweight.remote())
                #res.append(new_messages)
                count += 1
                #print("Worker at ",count)
                if not worker_tasks:
                    #print("Ended at ",count)
                    break

            #beta = res[-1]
            beta = ray.get(p.getweight.remote())
            #print("Estimated regression coefficients:", beta)
            #print("No. of iterations:", num_iter)
            #print("No. of test data:", ytest.shape[0])
            y_pred = pred_values(beta, xtest)
            #print("Correctly predicted labels:", np.sum(ytest == y_pred))
            #print("Accuracy", round(np.sum(ytest == y_pred)/len(ytest) * 100, 2), "%")

            #print("Accuracy: "+ np.sum(y == y_pred)/len(y) * 100)
            ax = round(np.sum(ytest == y_pred)/len(ytest) * 100, 2)
            acclist.append(ax)
            print(acclist)
            ttaken = time.time()-starttime
            timelist.append(ttaken)
            print(timelist)
            ray.shutdown()
            #print("Time Taken: ", time.time()-starttime)
        
        print("Mean Accuracy", statistics.mean(acclist))
        print("Mean Time", statistics.mean(timelist))
        # plt.scatter(acclist, timelist)
        # plt.show()
        '''
    
    
        # plotting regression line
        # plot_reg(X, y, beta)
        '''

