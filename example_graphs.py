#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nbujiashvili & sadler

Functions for generating adjacency matricies for graphs
"""

from inspect import GEN_SUSPENDED
import numpy as np
from snn_constants_equations import *
from jj_plot_functions import *
import math
import random

graph_types = ['trivial', 'WS', 'WS_SF', 'ER', 'SF', 'SPEC']

# Scale-free graph generator
def SF_adjacency_matrix(N, C, G_SF, seed_val):
    if(seed_val == 0):
        random_seed = np.random.randint(0, 2**32 - 1)
    else:
        random_seed = seed_val
    seed(random_seed)
    C = C*N
    M = np.empty((N,N), dtype=int)
    i = 0
    for i in np.arange(N):
        P = np.random.uniform(0, 0.66)
        K = round((C/P)**(1/G_SF))
        P = K/N
        for j in np.arange(N):
            if j < i:
                M[i][j] = M[j][i]
            elif i == j:
                M[i][j] = 0
            else:
                r = np.random.uniform(0, 1)
                if r > P:
                    M[i][j] = 0
                else:
                    M[i][j] = 1
    return M, random_seed


# Small-world scale-free graph generator
def WS_SF_adjacency_matrix(N, C, B, G_SF, seed_val):
    if(seed_val == 0):
        random_seed = np.random.randint(0, 2**32 - 1)
    else:
        random_seed = seed_val
    seed(random_seed)
    C = C*N
    M = np.zeros((N,N), dtype=int)
    i = 0
    while i < N:
        k = 1
        P = np.random.uniform(0, 0.66)
        K = round((C/P)**(1/G_SF))
        # print("Value fo C/P")
        # print(C/P)
        # print((C/P)**(1/G_SF))
        # print(N)
        if K > N:
            K = N - 1
        while k <= K/2:
            # usually connect right side
            r = np.random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                # print("GOT HERE")
                # print(N - K - 1)
                n = np.random.randint(0, math.ceil((N - K - 1)))
                # print("GOT There")
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
            k += 1
        # say K = 3 and beta = 0, half the time it will have two left connections and half the time it will only have one
        r = np.random.uniform(0, 1)
        if r < (K/2 + 1 - k):
            # usually connect right side
            r = np.random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                n = np.random.randint(0, math.ceil((N - K - 1)))
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
        i += 1  
    return M, random_seed

# Random graph
def ER_adjacency_matrix(N, density, seed_val):
    if(seed_val == 0):
        random_seed = np.random.randint(0, 2**32 - 1)
    else:
        random_seed = seed_val
    seed(random_seed)
    M = np.empty((N,N), dtype=int)
    for i in np.arange(N):
        for j in np.arange(N):
            if j < i:
                M[i][j] = M[j][i]
            elif i == j:
                M[i][j] = 0
            else:
                r = np.random.uniform(0, 1)
                if r > density:
                    M[i][j] = 0
                else:
                    M[i][j] = 1
    return M, random_seed

# Small-world graph generator
def WS_adjacency_matrix(N, K, B, seed_val):
    if(seed_val == 0):
        random_seed = np.random.randint(0, 2**32 - 1)
    else:
        random_seed = seed_val
    seed(random_seed)


    K = K*N
    M = np.zeros((N,N), dtype=int)
    i = 0
    while i < N:
        k = 1
        while k <= K/2:
            # usually connect right side
            r = np.random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                n = np.random.randint(0, math.ceil((N - K - 1)))
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
            k += 1
        # say K = 3 and beta = 0, half the time it will have two left connections and half the time it will only have one
        r = np.random.uniform(0, 1)
        if r < (K/2 + 1 - k):
            # usually connect right side
            r = np.random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                n = np.random.randint(0, math.ceil((N - K - 1)))
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
        i += 1  
    return M , random_seed

# trivial graphs
three_node = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
four_node = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
five_node = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
six_node = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
seven_node = np.array([[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0]])
trivial_graphs = [three_node, four_node, five_node, six_node, seven_node]