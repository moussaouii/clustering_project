import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

def load_config(pathConfig):
    '''
    Inputs :
        - pathConfig : path to json file that contain the configuration
    outputs :
        - python dictionary contain the config
    '''

    f = open(pathConfig)
    config = json.load(f)
    f.close()
    return config


def load_data(pathData,features):
    '''
    Inputs :
        - pathData : path to json file that contain data
        - features : a list of features name in the json file
    outputs :
        -  array of size (M*N) : M number of features and N number of data points
    '''
    data = pd.read_json(pathData)
    return data[features].values




def save_data(pathData,clustered_data):
    '''
    Inputs :
        - pathData : path to json file that contain data
        - clustered_data : a dataFrame containing the data and  additional features ('num_cluster')
    outputs :
        - results of clustring saved in csv 
    '''
    #write the results of clustring in csv 
    data = pd.read_json(pathData)
    data = data.merge(clustered_data,how='inner')
    data.to_csv('../output/results.csv',sep = ";")

    #save the visualization of the results by plotting the data colored by these labels. 
    plt.figure(figsize=(15,10))
    plt.scatter(data['longitude'], data['latitude'], c=data['num_cluster'], s=50, cmap='viridis')
    plt.title("result of clustring ")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig("../output/result_clustering.png")
