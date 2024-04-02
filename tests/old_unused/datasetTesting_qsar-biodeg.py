# imports
from pathlib import Path
import sys
path = Path(__file__)
sys.path.append(str(path.parent.parent))

from distance_classes import similarity_1d
from AntClust import AntClust
from rules import labroche_rules

#import matplotlib.pyplot as plt
import time as time
from scipy.io import arff
import sys
import numpy as np
import pandas as pd


def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())


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
    data = arff.loadarff('datasets/qsar-biodeg.arff')

    # load data as pandas data frame
    df = pd.DataFrame(data[0])

    # drop labels
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

    # normalize it via min_max_scaling
    for col in df.columns:
        df[col] = min_max_scaling(df[col])

    # or as a one liner
    # normalized_df = (df-df.min())/(df.max()-df.min())

    # save it as numpy array
    data_np = df.to_numpy()

    # normal numeric Distance, dataset is normalized so we know min = 0 max = 1
    min = 0
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
                         alpha_ant_meeting_iterations=150,
                         betta_template_init_meetings=0.5,
                         nest_shrink_prop=0.2)

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
    print('finding cluters')
    time_find_clusters = time.time()
    ant_clust.find_clusters()
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

    print('Sorted dictionary')
    print('-----------------------------')
    dic2 = dict(sorted(label_dic.items(), key=lambda x: x[1]))
    print(dic2)
