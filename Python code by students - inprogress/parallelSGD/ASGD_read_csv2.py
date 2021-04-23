import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cross_validation as cross_validation
import sklearn.linear_model as linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def loadCSV(filename):
    ''' 
    function to load dataset 
    '''
    # with open(filename, "r") as csvfile:
    #     lines = csv.reader(csvfile)
    #     dataset = list(lines)
    #     for i in range(len(dataset)):
    #         dataset[i] = [float(x) for x in dataset[i]]
    # return np.array(dataset)

    # Load local csv data into pandas DataFrame
    income_df = pd.read_csv("adult.csv")
    # income_df_list =  list(income_df)
    # print list(income_df)
    print income_df.shape