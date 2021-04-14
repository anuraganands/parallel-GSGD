import csv
import numpy as np


def loadCSV(filename):
    ''' 
    function to load dataset 
    '''
    with open(filename, "r") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    return np.array(dataset)
