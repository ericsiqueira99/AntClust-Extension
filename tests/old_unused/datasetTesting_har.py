# Just to test performance since it is a relatively big dataset (will take 12 minutes to run)
# imports
from pathlib import Path
import sys
path = Path(__file__)
sys.path.append(str(path.parent.parent))

from distance_classes import similarity_euclid2d, similarity_1d
from AntClust import AntClust
from rules import labroche_rules

import matplotlib.pyplot as plt
import time as time
from scipy.io import arff
import sys
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # har testing set

    # make set where last column=label is removed
    # transpose the whole thing and find max per feature
    # make similarity func for every feature and save it to array
    # run AntClust
    # compare results with the last item from the vector
    #for frame_index in range(len(data)):

    # -----------------------------------
    #        load dataset
    # -----------------------------------
    # this dataset will use eclidean distance for the first column and
    # normal distance for the second column
    data = arff.loadarff('datasets/hair.arff')
    # load data as pandas data frame
    df = pd.DataFrame(data[0])

    # delete labels
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

    # save it as numpy array
    data_np = df.to_numpy()
    #data_np = df.head(n=100).to_numpy()

    # -------------------------------
    #   define similarity objects
    # -------------------------------
    # transpose it
    #data_t = data_np.T
    #
    # print min, max
    #for index in range(len(data_t)):
    #    max = np.max(data_t[index])
    #    min = np.min(data_t[index])
    #    print(index, ': ', max, ' - ', min)
    #    print('--------------')
    #    print(data_t[index])

    # dataset har is normalized between -1 and 1 therefore we can simply make
    # our distance functions as below:
    # normal numeric Distance
    min = -1
    max = 1
    sim_numeric = similarity_1d(min, max)

    num_features = len(data_np[0])
    print('making similarity function vector')
    similarity_per_feature = [sim_numeric]*num_features

    # -----------------------------------
    #       define ruleset
    # -----------------------------------
    ruleset = labroche_rules()

    # -----------------------------------
    #       define AntClust
    # -----------------------------------
    print('Make Ant Colony')
    ant_clust = AntClust(data_np,
                         similarity_per_feature,
                         ruleset,
                         store_computed_similaritys=True,
                         number_meetings_template_initialization=5)

    print('-------------------------')
    print('Ant colony initialized')
    print('-------------------------')

    print('-------------------------')
    print('similarity Dictionary length')
    print('-------------------------')
    print(len(ant_clust.saved_similaritys))

    # -----------------------------------
    #       find Clusters
    # -----------------------------------
    print('finding cluters with iterations=', len(data_np)*6*4)
    time_find_clusters = time.time()
    ant_clust.find_clusters(len(data_np)*6*4)
    time_find_clusters = time.time() - time_find_clusters
    print('')
    print('Time to find Clusters')
    print('------------------------------')
    print(time_find_clusters)
    print()

    print('List of Labels:')
    for i in ruleset.labels:
        print(i)

    # hack for printing and visualize Clusters
    print('labels with ants and how many ants')
    label_dic = {}
    for ant in ant_clust.ants:
        if ant.label in label_dic:
            label_dic[ant.label] += 1
        else:
            label_dic[ant.label] = 1

    print('Clusters found: ', len(label_dic))
    print(label_dic)
