import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import logging
import sys
from sklearn.cluster import KMeans
from data_manager import load_config, load_data, save_data
from clustering import clustering_KMeans





def setup_logging(logPath):
    '''
    Inputs :
        - logPath : folder path where log files are generated
    outputs :
        - object that we used to logs in file and in stdout
    '''
    # set the format of log
    logFormatter = logging.Formatter("%(asctime)s::%(levelname)s::%(message)s")
    rootLogger = logging.getLogger()

    # add the fileHandler. The SfileHandler writes in the file 'logPath/<data>'
    fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # add the StreamHandler. The StreamHandler writes to stdout
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    # set the level of log
    rootLogger.setLevel(logging.DEBUG)
    return rootLogger





if len(sys.argv) > 1:
    pathConfig = sys.argv[1]
else:
    sys.stderr.write('Usage: python3 k_selection.py <path/to/config/file>\n ')
    sys.exit(1)
# configure the logging
log = setup_logging("../output")

# load the  configuration  
log.info('configuration has been loaded')
config=load_config(pathConfig)
nb_clusters = config['nb_clusters']
random_state = config['random_state']
pathData = config['pathData']
features = config['features']





# load the data 
data = load_data(pathData, features)
log.info('Data has been loaded')


# Training the model and get the clusters for different values of K
Sum_of_squared_distances = []
for k in range(1,20):
    results,model = clustering_KMeans(data,k,random_state)
    Sum_of_squared_distances.append(model.inertia_)
logging.info('Training has been done')

# Save a plot of elbrow

plt.plot(range(1,20), Sum_of_squared_distances, 'bx-')
plt.xlabel('k - nb of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For fing the Optimal k')
plt.grid(True)
plt.savefig("../output/elbrow_for_k_selection.png")








