# imports
from pathlib import Path
import sys
path = Path(__file__)
sys.path.append(str(path.parent))
import matplotlib.pyplot as plt
from distance_classes import similarity_euclid2d, similarity_1d
from AntClust import AntClust
from rules import labroche_rules
import time as time


if __name__ == '__main__':
    # Example AntClustCalling

    # -----------------------------------
    #        load dataset
    # -----------------------------------
    # this dataset will use eclidean distance for the first column and
    # normal distance for the second column
    data = [[[1, 1], 1],
            [[2, 2], 2],
            [[1, 2], 1],
            [[2, 1], 3],
            [[2, 3], 3],
            [[8, 8], 10],
            [[9, 9], 8],
            [[10, 10], 9]]

    # -------------------------------
    #   define similarity objects
    # -------------------------------
    # eclidean distance
    max = 14.142135623730951
    min = 0
    sim_vector = similarity_euclid2d(min, max)

    # Test
    print('Test eclidean distance')
    print('----------------------')
    print(sim_vector.similarity([0, 0], [0, 0]))
    print(sim_vector.similarity([0, 0], [5, 5]))
    print(sim_vector.similarity([0, 0], [10, 10]))

    # normal numeric Distance
    min = 0
    max = 10
    sim_numeric = similarity_1d(min, max)

    # Test
    print()
    print('Test numeric distance')
    print('----------------------')
    print(sim_numeric.similarity(0, 0))
    print(sim_numeric.similarity(0, 5))
    print(sim_numeric.similarity(0, 10))

    # define the similarity object array for AntClust
    # # FIXME: better description here
    similarity_per_feature = [sim_vector, sim_numeric]

    # -----------------------------------
    #       define ruleset
    # -----------------------------------
    ruleset = labroche_rules()

    # -----------------------------------
    #       define AntClust
    # -----------------------------------

    ant_clust = AntClust(similarity_per_feature,
                         ruleset)

    print('-------------------------')
    print('Ant templates after initialization')
    print('-------------------------')
    for i in range(len(ant_clust.ants)):
        print(ant_clust.ants[i].template)

    print('-------------------------')
    print('similarity Dictionary')
    print('-------------------------')
    print(ant_clust.saved_similaritys)

    # -----------------------------------
    #       find Clusters
    # -----------------------------------
    time_find_clusters = time.time()
    #ant_clust.find_clusters(300000)
    ant_clust.fit(data)
    time_find_clusters = time.time() - time_find_clusters
    print('')
    print('Time to find Clusters')
    print('------------------------------')
    print(time_find_clusters)
    print()

    # get Clusters
    clusters_indexes = ant_clust.get_clusters()

    print('ClusterIndexes')
    print('-----------------')
    print(clusters_indexes)

    # hack for printing and visualize Clusters
    cluster_data_dict = {}
    for i in ruleset.labels:
        cluster_data_dict[i] = []

    i = 0
    for ant in ant_clust.ants:
        print('Ant', i, ':', ant.label)
        cluster_data_dict[ant.label].append(ant.gene)
        i += 1

    x0 = []
    y0 = []
    for item in cluster_data_dict[0]:
        x0.append(item[0][0])
        y0.append(item[0][1])

    x1 = []
    y1 = []
    for item in cluster_data_dict[1]:
        x1.append(item[0][0])
        y1.append(item[0][1])

    if len(cluster_data_dict) > 2:
        x2 = []
        y2 = []
        for item in cluster_data_dict[2]:
            x2.append(item[0][0])
            y2.append(item[0][1])

        plt.scatter(x2, y2, c="black",
                    linewidths=2,
                    marker=",",
                    edgecolor="black",
                    s=200)

    plt.scatter(x0, y0, c="pink",
                linewidths=2,
                marker="s",
                edgecolor="green",
                s=50)

    plt.scatter(x1, y1, c="yellow",
                linewidths=2,
                marker="^",
                edgecolor="red",
                s=200)

    plt.show()
