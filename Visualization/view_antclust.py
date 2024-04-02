"""
Module to allow visualization of ANTCLUST
"""

import pandas as pd
from itertools import cycle
from sklearn.metrics import adjusted_rand_score, rand_score
import matplotlib.pyplot as plt


def visualize_meetings(cluster_evolution, rule_applied, labels):
    ari_evolution = [adjusted_rand_score(labels, cluster) for cluster in cluster_evolution]
    meetings = range(1,len(ari_evolution)+1)
    data = {'Meeting number': meetings, 'Rule applied': rule_applied, 'ARI': ari_evolution}
    df = pd.DataFrame(data)
    # Plotting
    plt.figure(figsize=(8, 6))

    # Iterate through each rule and plot its points
    for rule, group in df.groupby('Rule applied'):
        if rule != 'R6': 
            plt.scatter(group['Meeting number'], group['ARI'], label=rule, alpha=0.8)
        else:
            pass
            # plt.scatter(group['Meeting number'], group['ARI'], label=rule, alpha=0.1)  


    # Add labels and legend
    plt.xlabel('Meeting Number')
    plt.ylabel('ARI')
    plt.title('AntClust Evolution')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

def visualize_meetings_subplots(cluster_evolution, rule_applied, labels):
    ari_evolution = [adjusted_rand_score(labels, cluster) for cluster in cluster_evolution]
    meetings = range(1,len(ari_evolution)+1)
    data = {'Meeting number': meetings, 'Rule applied': rule_applied, 'ARI': ari_evolution}
    df = pd.DataFrame(data)
    # Plotting
    plt.figure(figsize=(16, 8))

    # Get unique rules and assign colors
    unique_rules = df['Rule applied'].unique()
    colors = cycle(['blue', 'green', 'red', 'orange', 'purple', 'brown'])

    # Plotting subplots for each rule
    fig, axs = plt.subplots(len(unique_rules), 1, figsize=(8, 6), sharex=True)

    for i, rule in enumerate(unique_rules):
        rule_data = df[df['Rule applied'] == rule]
        axs[i].scatter(rule_data['Meeting number'], rule_data['ARI'], color=next(colors), alpha=0.5)
        axs[i].set_title(rule)
        axs[i].set_ylabel('ARI')

    # Add common x-axis label
    plt.xlabel('Meeting Number')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
