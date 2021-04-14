import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from ASGD_logistic_func import logistic_func, log_gradient, cost_func, grad_desc, pred_values
import numpy as np
from multiprocessing.connection import wait
import ray

@ray.remote
class ParameterServer(object):

    def __init__(self, x, y, b):
        self.x = x
        self.y = y
        self.b = b

    def run(self):
        print('Parameter Server Started')
              

    def startSlaves(self):

        return self.b

    def updatewight(self, x):
        self.b = x
    
    def getweight(self):
        return self.b
    
