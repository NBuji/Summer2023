#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nbujiashvili & sadler

Functions for the SNN
This is honestly where everything interesting is going to be happening
The rest of the code is fixed as in it is clear what it should be
This is what will need to be tweaked and what determines annealing times
"""

import numpy as np
import sys
from snn_constants_equations import *
import csv

# create a q matrix for the SNN's synapses
def get_q_matrix(A):
    N = A.shape[0]
    Q = np.empty((A.shape), dtype=int)
    for i in np.arange(N):
        for j in np.arange(N):
            if i == j:
                Q[i][i] = (beta * np.sum(A[i]) - alpha * (N - 1))
            else:
                if A[i][j] == 0:
                    Q[i][j] = alpha
                elif A[i][j] == 1:
                    Q[i][j] = alpha - beta
    return Q

# create a negative q matrix for the SNN's synapses
def get_negative_q_matrix(A):
    N = A.shape[0]
    Q = np.empty((A.shape), dtype=int)
    for i in np.arange(N):
        for j in np.arange(N):
            if i == j:
                Q[i][i] = -(beta * np.sum(A[i]) - alpha * (N - 1))
            else:
                if A[i][j] == 0:
                    Q[i][j] = -alpha
                elif A[i][j] == 1:
                    Q[i][j] = beta - alpha
    return Q

# scale Q matrix appropriately
def scale_Q(Q, my_parent_path):
    N = Q.shape[0]
    q_iis = np.zeros(N)
    for i in np.arange(N):
        q_iis[i] = abs(Q[i][i])
    min = np.amin(q_iis)
    if min == 0:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("all nodes are connected to eachother making choice of partition arbitrary \n")
            f.close()
        sys.exit()
    scale_factor = min_height/min
    scaled_Q = Q*scale_factor
    return scaled_Q

# calculate bias currents, spike hieghts, and voltage thresholds
def get_bias_and_thresholds(Q,Spike_Num):
    N = Q.shape[0]
    
    spike_heights = np.empty(N, dtype=float)
    for i in np.arange(N):
        spike_heights[i] = Q[i][i]

    # min_th_spike_height is the minimum spike height required for a JJ neuron
    # I think I want to scale it by the max - min
    # I think the lambda values might be okay as they are, that way we never need to change new neurons
    # Let's do some testing on these labda values and see how we can scale it

    spike_hieght_min = np.amin(spike_heights)
    spike_height_spread = np.amax(spike_heights) - spike_hieght_min
    bias_currents = np.empty(N, dtype=float)
    for i in np.arange(N):
        #bias_currents[i] = bc_sh_intercept + bc_sh_gradient * ((min_th_spike_height/min_height)*spike_heights[i])
        # bias_currents[i] = 1.7 + ((spike_heights[i] - spike_hieght_min)/spike_height_spread)*0.2
        # inp = spike_heights[i]
        # bias_currents[i] = 1.7962 + 0.9227*inp + (-2.5928)*(inp**2) + 2.412*(inp**3) + (-0.8705)*(inp**4)
        bias_currents[i] = get_matrix_bias_current(spike_heights[i], Spike_Num)
    threshold_voltages = np.empty(N, dtype=float)
    for i in np.arange(N):
        threshold_voltages[i] = bias_currents[i]/1.9

    return bias_currents, spike_heights, threshold_voltages

# annealing scales with N^{2} and must be greater than 1000 tau
def get_simulation_time(N):
    # simulation_time = 100*N
    # relationship between N and time derived by experimenting with optimal setteling times
    simulation_time = 77*(N**2)
    if simulation_time < 1000: # simulation time can't be too short otherwise this is just ridiculous
        simulation_time = 1000
    
    return simulation_time

    # Retrieving the input current vs bias current 
def get_spike_matrix():
    S_M_1 = []
    file_name = 'S_M_40_2.csv'
    mat_dim = 0 #matrix dimensions
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            S_M_1.append(row) 
            mat_dim = mat_dim +1   
    S_M = np.zeros((mat_dim,mat_dim), dtype=float)
     
    for i in range(0,mat_dim):
        for j in range(0,mat_dim):
              S_M[i][j] = S_M_1[i][j]
    return S_M

    # Retrieving the bias current corresponding to the input input current 
def get_matrix_bias_current(spike_h, spike_num_1):
    M_1 = get_spike_matrix()
    row_n, column_n = M_1.shape
    input_current_marks = np.linspace(0, 1, row_n )
    bias_current_marks = np.linspace(1.6,1.9, column_n)
    input_current_diff = np.linspace(0, 1, row_n )
    for i in range(row_n):
        input_current_diff[i] = abs(input_current_marks[i] - spike_h)
    min_cur = np.min(input_current_diff)
    min_cur_ind = np.argmin(input_current_diff)
    fin_ind = 0
    while((M_1[fin_ind][min_cur_ind] < spike_num_1) and (fin_ind < row_n -1)):
        fin_ind = fin_ind + 1
    return bias_current_marks[fin_ind]

#Not finished: attempt at retrieving the edge line in the input vs bias current matrix 
def find_count_line(spike_count):
    M_1 = get_spike_matrix()
    row_n, column_n = M_1.shape
    input_current_marks = np.linspace(0, 1, column_n )
    bias_current_marks = np.linspace(1.6,1.9, row_n)
    line_points = []
    for i in range(column_n):
        j = 0
        
        while( j < row_n and M_1[j][i] < spike_count):
            j = j +1
        if(M_1[j][i] >= spike_count):
            line_points.append(())