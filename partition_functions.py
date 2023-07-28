#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sadler

A functions related to partitions and cut sizes
"""

import numpy as np
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
import itertools
from networkx.algorithms.community.quality import modularity
import matplotlib.pyplot as plt
from snn_constants_equations import *

# returns the communities from a girvan_newman parition
def get_girvan_newman_partition(G):
    comp = girvan_newman(G)
    first_communities = tuple(sorted(c) for c in next(itertools.islice(comp, 1)))
    return first_communities

# get partition s from spiking behaviour
def get_s(array):
    N = array.shape[2]
    neuron_spikes = array[-1][0] - array[-2][0]
    s = np.zeros(N, dtype=int)
    for n in np.arange(N):
        # add a statement in here seeing if the number of spikes is far from the average. If not, make a statement about it and maybe go back to annealing for longer.
        # looks like we should have at least 10 spikes in 200 tau for a 1
        # less than 3 spikes should indicate a -1
        # anything inbetween that seems like it should be inconclusive
        if neuron_spikes[n] >= 10:
            s[n] = 1
        else:
            s[n] = -1
    return s

# get partition x from spiking behaviour
def get_x(array):
    N = array.shape[2]
    neuron_spikes = array[-1][0] - array[-2][0]
    x = np.empty(N, dtype=int)
    for n in np.arange(N):
        # add a statement in here seeing if the number of spikes is far from the average. If not, make a statement about it and maybe go back to annealing for longer.
        # looks like we should have at least 10 spikes in 200 tau for a 1
        # less than 3 spikes should indicate a -1
        # anything inbetween that seems like it should be inconclusive
        if neuron_spikes[n] >= 10:
            x[n] = 1
        else:
            x[n] = 0
    return x

# get partition in the form of x from s
def get_x_from_s(s):
    x = [0.5*(a - 1) for a in s]
    return x

# get partition in the form of s from x
def get_s_from_x(x):
    s = [2*a - 1 for a in x]
    return s

# return partition in the form of x from communities
def get_x_from_communities(coms):
    length = 0
    for c in coms:
        length += len(c)
    x = np.zeros(length, dtype=int)
    for i in np.arange(length):
        if i in coms[0]:
            x[i] = 1
    return x

# return communities from partition in the form of x
def get_communities_from_x(x):
    coms = [[], []]
    for i in np.arange(x.size):
        if x[i] == 0:
            coms[0].append(i)
        else:
            coms[1].append(i)
    return coms

# function which finds the next partition of a graph which will go through all possible partitions
def permute_s(s):
    indexes_positive = np.where(s == 1)[0]
    if indexes_positive.size == 0:
        s[0] = 1
        return s
    else:
        if ((indexes_positive[-1] + 1) != s.size):
            s[indexes_positive[-1] + 1] = 1
            if np.where(s == -1)[0].size == 0:
                s[-2] = -1
                return s
        else:
            if (np.where(s == -1)[0][-1] + 1) == (s.size - 1):
                s[indexes_positive[-2]] = -1
                s[indexes_positive[-2] + 1] = 1
                for j in range((indexes_positive[-2] + 2), (s.size)):
                    s[j] = -1
                return s
            else:
                s[-2] = -1
                return s  
    return s

# function which finds the next partition of a graph which will go through all possible partitions
def permute_x(x):
    indexes_positive = np.where(x == 1)[0]
    if indexes_positive.size == 0:
        x[0] = 1
        return x
    else:
        if ((indexes_positive[-1] + 1) != x.size):
            x[indexes_positive[-1] + 1] = 1
            if np.where(x == 0)[0].size == 0:
                x[-2] = 0
                return x
        else:
            if (np.where(x == 0)[0][-1] + 1) == (x.size - 1):
                x[indexes_positive[-2]] = 0
                x[indexes_positive[-2] + 1] = 1
                for j in range((indexes_positive[-2] + 2), (x.size)):
                    x[j] = 0
                return x
            else:
                x[-2] = 0
                return x 
    return x

# get energy of partition
def get_energy(negQ, x):
    N = x.size
    energy = 0
    for i in np.arange(N):
        energy += negQ[i][i]*x[i]
        j = i + 1
        while j < N:
            energy += negQ[i][j]*x[i]*x[j]
            j += 1
    return energy

# get cut size
def get_cut_size(A, s):
    cut_size = np.matmul(np.matmul(s, (np.subtract((np.ones(A.shape, dtype=int))*alpha, A*beta))), s)
    return cut_size

# returns the cut, energy, and modularity of a parition of a graph
def get_cut_energy_modularity(A, negQ, G, x, s, coms):
    cut = get_cut_size(A, s)
    energy = get_energy(negQ, x)
    mod = modularity(G, coms)
    return cut, energy, mod

# returns the max energy of a graph using the q matrix
def get_ideal_cut_energy_modularity(A, negQ, G):
    x = np.full(A.shape[0], 0, dtype = int)
    s = get_s_from_x(x)
    coms = get_communities_from_x(x)
    ideal_cut, ideal_energy, ideal_modularity = get_cut_energy_modularity(A, negQ, G, x, s, coms)
    ideal_cut_x = x.copy()
    ideal_energy_x = x.copy()
    ideal_modularity_x = x.copy()
    ideal_cut_cut = ideal_energy_cut = ideal_modularity_cut = ideal_cut
    ideal_cut_energy = ideal_energy_energy = ideal_modularity_energy = ideal_energy
    ideal_cut_modularity = ideal_energy_modularity = ideal_modularity_modularity = ideal_modularity
    N = negQ.shape[0]
    total_num_partitions = 2**(N-1) - 1
    i = 0
    while i < total_num_partitions:
        x = permute_x(x)
        s = get_s_from_x(x)
        coms = get_communities_from_x(x)
        cut, energy, mod = get_cut_energy_modularity(A, negQ, G, x, s, coms)
        if (cut < ideal_cut_cut) or ((cut == ideal_cut_cut) and (energy >= ideal_cut_energy) and (mod >= ideal_cut_modularity)):
            ideal_cut_x = x.copy()
            ideal_cut_cut, ideal_cut_energy, ideal_cut_modularity = cut, energy, mod
        if (energy > ideal_energy_energy) or ((cut <= ideal_energy_cut) and (energy == ideal_energy_energy) and (mod >= ideal_energy_modularity)):
            ideal_energy_x = x.copy()
            ideal_energy_cut, ideal_energy_energy, ideal_energy_modularity = cut, energy, mod
        if (mod > ideal_modularity_modularity) or ((cut <= ideal_modularity_cut) and (energy >= ideal_modularity_energy) and (mod == ideal_modularity_modularity)):
            ideal_modularity_x = x.copy()
            ideal_modularity_cut, ideal_modularity_energy, ideal_modularity_modularity = cut, energy, mod
        i += 1
    return ideal_cut_x, ideal_energy_x, ideal_modularity_x

# plots partitioned graph. partition can be either in the for of x or s
def plot_partition(G, partition, title, my_path, graph_layout):
    N = partition.size

    colors = ['red', 'cyan']
    color_map = np.empty(N, dtype=str)
    for i in np.arange(N):
        if partition[i] == 1:
            color = colors[1]
        else:
            color = colors[0]
        color_map[i] = color
    
    plt.figure()
    if(graph_layout == 0):
        layout_position = nx.circular_layout(G)
        nx.draw(G, node_color=color_map, with_labels=True, pos= layout_position)
    elif(graph_layout == 1):
        layout_position = nx.random_layout(G)
        nx.draw(G, node_color=color_map, with_labels=True, pos= layout_position)
    elif(graph_layout == 2):
        layout_position = nx.shell_layout(G)
        nx.draw(G, node_color=color_map, with_labels=True, pos= layout_position)
    elif(graph_layout == 3):
        layout_position = nx.spectral_layout(G)
        nx.draw(G, node_color=color_map, with_labels=True, pos= layout_position)
    else:
        nx.draw(G, node_color=color_map, with_labels=True)
    plt.savefig(f'{my_path}/{title}.png')
    plt.clf()
    plt.close()
    return