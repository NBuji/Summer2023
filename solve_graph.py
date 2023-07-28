#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nbujiashvili & sadler

solving all graph types
code Summer 2023

"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
import metis
import sys
import os
import time
import re
from si_prefix import si_format
from snn_constants_equations import *
from snn_functions import *
from example_graphs import *
from snn_functions import *
from gpp_snn import *
from partition_functions import *
from time_factor_input import *
import csv


seed_val = 0
if len(sys.argv) < 3:
    print("Insufficient parameters")
    sys.exit()

graph_type = sys.argv[1] # graph type (trivial, WS, WS_SF, ER, SF)
N = int(sys.argv[2]) # number of nodes
from datetime import datetime
dt_now = datetime.now() # datetime object containing current date and time

# check for valid graph type
if graph_type not in graph_types:
    print("Invalid graph type has been selected")
    sys.exit()

graph_name = f'{N}_{graph_type}'
# append graph name if already exists
my_parent_path = f'{save_to}{graph_name}'
while os.path.exists(my_parent_path):
    match = re.search(r'\d+$', my_parent_path)
    if match:
        # Get the numeric portion of the string
        num_str = match.group()
        # Increment the numeric portion of the string
        num = int(num_str) + 1
        # Get the non-numeric portion of the string
        prefix = my_parent_path[:match.start()]
        # Concatenate the non-numeric and incremented numeric portion of the string
        my_parent_path =  f"{prefix}{num:0{len(num_str)}d}"
    else:
        # If the string does not end with a numeric portion, simply append "1"
        my_parent_path =  f"{my_parent_path}_1"
os.makedirs(my_parent_path)

# optional arguments
num_runs = 1
extra_args = 0
for a in sys.argv:
    if a[:5] == 'runs=':
        if a[5:].isnumeric:
            num_runs = int(a[5:])
        else:
            with open(f'{my_parent_path}/information.txt', 'a') as f:
                f.write("typo in runs \n")
                f.close()
            sys.exit()
        if num_runs < 1:
            with open(f'{my_parent_path}/information.txt', 'a') as f:
                f.write("need atleast 1 run1 \n")
                f.close()
            sys.exit()
        extra_args += 1

# general exception
if N < 2:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("the number of nodes is less than 4. This graph is meaningless \n")
            f.close()
        sys.exit()

# scale free graphs
if graph_type == 'SF':
    if len(sys.argv) != 6 + extra_args:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("Insufficient parameters In the Input \n")
            f.close()
        sys.exit()

    clustering = float(sys.argv[3]) # clustering constant K
    gamma_SF = float(sys.argv[4])
    seed_val = float(sys.argv[6]) #Determinig seed value for the simulation
    seed_val = int(seed_val)

    # exceptions
    if (clustering < 0) or (gamma_SF < 0):
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("inputs must all be positive \n")
            f.close()
        sys.exit()
    if clustering > 1:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("clustering is too high for a small world graph \n")
            f.close()
        sys.exit()
    if gamma_SF < 2:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("gamma value is too small for a scale free graph \n")
            f.close()
        sys.exit()
    if gamma_SF > 3:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("gamma value is too large for a scale free graph \n")
            f.close()
        sys.exit()

    A, recorded_seed = SF_adjacency_matrix(N, clustering, gamma_SF, seed_val) # returns an adjacency matrix

# Small world scale free graphs
if graph_type == 'WS_SF':
    if len(sys.argv) != 7 + extra_args:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("Insufficient parameters In WS_SF Input \n")
            f.close()
        sys.exit()

    clustering = float(sys.argv[3]) # clustering constant K
    probability = float(sys.argv[4]) # probability of rogue connections
    gamma_SF = float(sys.argv[5])
    seed_val = float(sys.argv[7]) #Determinig seed value for the simulation
    seed_val = int(seed_val)

    # exceptions
    if (clustering < 0) or (probability < 0) or (gamma_SF < 0):
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("inputs must all be positive \n")
            f.close()
        sys.exit()
    if clustering > 1:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("clustering is too high for a small world graph \n")
            f.close()
        sys.exit()
    if probability > 1:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("probability must be less than 1 \n")
            f.close()
        sys.exit()
    if (probability < 0.0005) or (probability > 2):
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("probability is outside the bounds of what a small world graph really is but it's okay ;) \n")
            f.close()
    if gamma_SF < 2:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("gamma value is too small for a scale free graph \n")
            f.close()
        sys.exit()
    if gamma_SF > 3:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("gamma value is too large for a scale free graph \n")
            f.close()
        sys.exit()

    A, recorded_seed = WS_SF_adjacency_matrix(N, clustering, probability, gamma_SF, seed_val) # returns an adjacency matrix and a seed value for the corresponding code

# Small world graphs
elif graph_type == 'WS':
    if len(sys.argv) != 6 + extra_args:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("Insufficient parameters \n")
            f.close()
        sys.exit()

    clustering = float(sys.argv[3]) # clustering constant K
    probability = float(sys.argv[4]) # probability of rogue connections
    seed_val = float(sys.argv[6]) #Determinig seed value for the simulation
    seed_val = int(seed_val)

    # exceptions
    if (clustering < 0) or (probability < 0):
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("inputs must all be positive \n")
            f.close()
        sys.exit()
    if clustering > 1:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("clustering cannot be greater than 1 \n")
            f.close()
        sys.exit()
    if probability > 1:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("probability must be less than 1 \n")
            f.close()
        sys.exit()
    if (probability < 0.0005) or (probability > 0.2):
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("probability is outside the bounds of what a small world graph really is but it's okay ;) \n")
            f.close()

    A, recorded_seed = WS_adjacency_matrix(N, clustering, probability, seed_val) #this is always an adjacency matrix

# Random graphs
elif graph_type == 'ER':
    if len(sys.argv) != 5  + extra_args:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("Insufficient parameters \n")
            f.close()
        sys.exit()

    density = float(sys.argv[3]) # density of connections
    seed_val = float(sys.argv[5]) #Determinig seed value for the simulation
    seed_val = int(seed_val)

    # exceptions
    if (density < 0):
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("inputs must all be positive \n")
            f.close()
        sys.exit()
    if density > 1:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("density must be less than 1 \n")
            f.close()
        sys.exit()
    
    A, recorded_seed = ER_adjacency_matrix(N, density, seed_val) #this is always an adjacency matrix

# Retrieving the adjacency matrix from the specific csv file
elif graph_type == 'SPEC':
    
    A_1 = []
    file_name = csv_file_entry()
    mat_dim = 0 #matrix dimensions
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            A_1.append(row) 
            mat_dim = mat_dim +1   
    A = np.zeros((mat_dim,mat_dim), dtype=int)
    recorded_seed = 0   
    for i in range(0,mat_dim):
        for j in range(0,mat_dim):
            if(int((A_1[i][j])[0]) > 0):
                A[i][j] = 1    

# Trivial graphs
elif graph_type == 'trivial':
    if len(sys.argv) != 3  + extra_args:
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("Insufficient parameters \n")
            f.close()
        sys.exit()
    
    # exceptions
    if (N < 3) or (N > 7):
        with open(f'{my_parent_path}/information.txt', 'a') as f:
            f.write("the number of nodes is outside what I have trivial graphs for \n")
            f.close()
        sys.exit()

    A = trivial_graphs[N - 3]

# Now that we have our adjacency matrix for the graph, let's analyze the graph and see how we can best partition it!
negQ = get_negative_q_matrix(A)
Q = get_q_matrix(A)
scaled_negQ = scale_Q(negQ, my_parent_path)
simulation_time = get_simulation_time(N) #how long the simulation needs to run for
time_factor_1, sigma_factor_1, graph_type_1, stoch_factor = get_time_factor()
simulation_time = simulation_time * time_factor_1

edge_spike_count = 11
bias_currents, spike_heights, threshold_voltages = get_bias_and_thresholds(scaled_negQ,edge_spike_count )

# Creates a frequency plot of bias current values
def bias_currents_frequency_plot(bias_currents):
    unique_values, counts = np.unique(bias_currents, return_counts=True)

    distinct_values = 0
    for i, bias in enumerate(bias_currents):
        distinct_values += 1
        i -= 1
        while i >= 0:
            if bias_currents[i] == bias:
                distinct_values -= 1
                i = -1
            i -= 1


    if not os.path.exists(f'{my_parent_path}'):
        os.makedirs(f'{my_parent_path}')
    plt.figure()

    # Create a bar plot of the counts
    sns.histplot(bias_currents, bins=len(unique_values), kde=False)

    # Set plot title and labels
    plt.title(f"{distinct_values} Unique Bias Currents")
    plt.xlabel("Bias Current")
    plt.ylabel("Frequency")

    plt.savefig(f'{my_parent_path}/bias_current_frequencies.png')
    plt.clf()
    plt.close()
    return
bias_currents_frequency_plot(bias_currents)

# Write information about this graph and the simulation
with open(f'{my_parent_path}/information.txt', 'a') as f:
    f.write(f'''Graph information for {graph_name}
Simulation started on {dt_now} \n''')
    f.write('Arguments:\npython3 ')
    for ar in sys.argv:
        f.write(f'{ar} ')
    f.write(f'''\nnumber of nodes is: {N}
simulation time is: {simulation_time} tau
\nConstants: \n\tgamma: {gamma} \n\tsmall_lambda: {small_lambda} \n\tlambda_s: {lambda_s} \n\tlambda_p: {lambda_p} \n\teta: {eta} \n\tstochasticity: {stochasticity} \n\tdelay_duration: {delay_duration} \n\tpulse_width: {pulse_width} \n\talpha: {alpha} \n\tbeta: {beta} \n\tmin_height: {min_height} \n\tslope_change: {slope_change} \n\tbc_sh_gradient: {bc_sh_gradient} \n\tbc_sh_intercept: {bc_sh_intercept} \n\tmin_th_spike_height: {min_th_spike_height} \n''')
    f.close()

# create csv files about this graph
if not os.path.exists(f'{my_parent_path}/csv'):
    os.makedirs(f'{my_parent_path}/csv')
np.savetxt(f'{my_parent_path}/csv/A.csv', A, delimiter=',')
np.savetxt(f'{my_parent_path}/csv/Q.csv', Q, delimiter=',')
np.savetxt(f'{my_parent_path}/csv/negQ.csv', negQ, delimiter=',')
np.savetxt(f'{my_parent_path}/csv/scaled_negQ.csv', scaled_negQ, delimiter=',')
np.savetxt(f'{my_parent_path}/csv/bias_currents.csv', bias_currents, delimiter=',')
np.savetxt(f'{my_parent_path}/csv/spike_heights.csv', spike_heights, delimiter=',')
np.savetxt(f'{my_parent_path}/csv/threshold_voltages.csv', threshold_voltages, delimiter=',')

G = nx.from_numpy_array(A) # create graph in the form of NetworkX

# Benchmark Partitions
metis_start = time.time()
metis_x_list = metis.part_graph(G, 2)[1]
metis_end = time.time()
metis_x = np.array(metis_x_list, dtype=int) # changing to an np array for ease of use
time_metis = metis_end - metis_start
metis_s = get_s_from_x(metis_x)
metis_coms = get_communities_from_x(metis_x)

kernighan_lin_start = time.time()
kernighan_lin_coms = community.kernighan_lin.kernighan_lin_bisection(G)
kernighan_lin_end = time.time()
time_kernighan_lin = kernighan_lin_end - kernighan_lin_start
kernighan_lin_x = get_x_from_communities(kernighan_lin_coms)
kernighan_lin_s = get_s_from_x(kernighan_lin_x)

girvan_newman_start = time.time()
girvan_newman_coms = get_girvan_newman_partition(G)
girvan_newman_end = time.time()
time_girvan_newman = girvan_newman_end - girvan_newman_start
if len(girvan_newman_coms) > 2:
    with open(f'{my_parent_path}/information.txt', 'a') as f:
        f.write(f'Girvan_Newman found more than 2 partitions!!!\nIgnore these results!!!\n')
        f.close()
girvan_newman_x = get_x_from_communities(girvan_newman_coms)
girvan_newman_s = get_s_from_x(girvan_newman_x)

metis_cut, metis_energy, metis_modularity = get_cut_energy_modularity(A, negQ, G, metis_x, metis_s, metis_coms)
kernighan_lin_cut, kernighan_lin_energy, kernighan_lin_modularity = get_cut_energy_modularity(A, negQ, G, kernighan_lin_x, kernighan_lin_s, kernighan_lin_coms)
girvan_newman_cut, girvan_newman_energy, girvan_newman_modularity = get_cut_energy_modularity(A, negQ, G, girvan_newman_x, girvan_newman_s, girvan_newman_coms)

plot_partition(G, metis_x, "metis_partition", my_parent_path, graph_type_1)
plot_partition(G, kernighan_lin_x, "kernighan_lin_partition", my_parent_path, graph_type_1)
plot_partition(G, girvan_newman_x, "girvan_newman_partition", my_parent_path, graph_type_1)

with open(f'{my_parent_path}/information.txt', 'a') as f:
    f.write(f'''\nBenchmark Partitions:
    Partition Type & Cut & Energy & Modularity
    \\multirow{{{num_runs + 6}}}{{*}}{{{N}}} & \\rowcolor{{yellow}} metis & {si_format(time_metis, precision=2)}s  & {metis_cut} & {metis_energy} & {round(kernighan_lin_modularity, -int(math.floor(math.log10(abs(kernighan_lin_modularity)))) + 2)}  \\\\
     & \\rowcolor{{yellow}} kernighan-lin & {si_format(time_kernighan_lin, precision=2)}s & {kernighan_lin_cut} & {kernighan_lin_energy} & {round(kernighan_lin_modularity, -int(math.floor(math.log10(abs(kernighan_lin_modularity)))) + 2)} \\\\
     & \\rowcolor{{yellow}} girvan-newman & {si_format(time_girvan_newman, precision=2)}s & {girvan_newman_cut} & {girvan_newman_energy} & {round(girvan_newman_modularity, -int(math.floor(math.log10(abs(girvan_newman_modularity)))) + 2)}  \\\\ \n''')
    f.close()

# Calculate ideal partitions
def get_ideal_partitions():
    start_ideal = time.time()
    ideal_cut_x, ideal_energy_x, ideal_modularity_x = get_ideal_cut_energy_modularity(A, negQ, G)
    end_ideal = time.time()
    time_ideal = end_ideal - start_ideal

    plot_partition(G, ideal_cut_x, "ideal_cut_partition", my_parent_path, graph_type_1)
    plot_partition(G, ideal_energy_x, "ideal_energy_partition", my_parent_path, graph_type_1)
    plot_partition(G, ideal_modularity_x, "ideal_modularity_partition", my_parent_path, graph_type_1)
    
    ideal_cut_s = get_s_from_x(ideal_cut_x)
    ideal_cut_coms = get_communities_from_x(ideal_cut_x)
    ideal_cut_cut, ideal_cut_energy, ideal_cut_modularity = get_cut_energy_modularity(A, negQ, G, ideal_cut_x, ideal_cut_s, ideal_cut_coms)
    ideal_energy_s = get_s_from_x(ideal_energy_x)
    ideal_energy_coms = get_communities_from_x(ideal_energy_x)
    ideal_energy_cut, ideal_energy_energy, ideal_energy_modularity = get_cut_energy_modularity(A, negQ, G, ideal_energy_x, ideal_energy_s, ideal_energy_coms)
    ideal_modularity_s = get_s_from_x(ideal_modularity_x)
    ideal_modularity_coms = get_communities_from_x(ideal_modularity_x)
    ideal_modularity_cut, ideal_modularity_energy, ideal_modularity_modularity = get_cut_energy_modularity(A, negQ, G, ideal_modularity_x, ideal_modularity_s, ideal_modularity_coms)
    
    with open(f'{my_parent_path}/information.txt', 'a') as f:
        f.write(f'''\nIdeal Partitions:
    Partition Type & Cut & Energy & Modularity
     & \\rowcolor{{cyan}} ideal cut & {si_format(time_ideal, precision=2)}s & {ideal_cut_cut} & {ideal_cut_energy} & {round(ideal_cut_modularity, -int(math.floor(math.log10(abs(ideal_cut_modularity)))) + 2)}  \\\\
     & \\rowcolor{{cyan}} ideal energy & {si_format(time_ideal, precision=2)}s & {ideal_energy_cut} & {ideal_energy_energy} & {round(ideal_energy_modularity, -int(math.floor(math.log10(abs(ideal_energy_modularity)))) + 2)}  \\\\
     & \\rowcolor{{cyan}} ideal modularity & {si_format(time_ideal, precision=2)}s & {ideal_modularity_cut} & {ideal_modularity_energy} & {round(ideal_modularity_modularity, -int(math.floor(math.log10(abs(ideal_modularity_modularity)))) + 2)}  \\\\ \n''')
        f.close()
    return

# If the graph is small and ideal partitions will not take long to calculate, run them before SNN partitions
if N < 6:
    get_ideal_partitions()

# SNN partitioning
with open(f'{my_parent_path}/information.txt', 'a') as f:
    f.write(f'''\nSNN Partitions:
    Partition Type & Cut & Energy & Modularity & Actual Time \n''')
    f.close()

for run in np.arange(num_runs):
    my_path = f'{my_parent_path}/{run}'
    os.makedirs(my_path)
    with open(f'{my_path}/information.txt', 'a') as f:
        f.write(f"Information for run number {run} \n")
        f.close()
    
    snn_actual_start = time.time()
    spike_arrays = get_SNN_partition(scaled_negQ, simulation_time, bias_currents, spike_heights, threshold_voltages, my_path, recorded_seed, sigma_factor_1, time_factor_1, stoch_factor)
    snn_actual_end = time.time()
    time_snn_actual = snn_actual_end - snn_actual_start

    snn_x = get_x(spike_arrays)
    plot_partition(G, snn_x, "partitioned_graph", my_path, graph_type_1)
    #np.savetxt(f'{my_path}/csv/final_partition.csv', snn_x, delimiter=',')
    snn_coms = get_communities_from_x(snn_x)
    snn_s = get_s_from_x(snn_x)
    snn_cut, snn_energy, snn_modularity = get_cut_energy_modularity(A, negQ, G, snn_x, snn_s, snn_coms)
    time_snn = get_simulation_time(N)*6*10**-13
    with open(f'{my_parent_path}/information.txt', 'a') as f:
        f.write(f'\t & \\rowcolor{{pink}} SNN \#{run} & {si_format(time_snn, precision=2)}s & {snn_cut} & {snn_energy} & {round(snn_modularity, -int(math.floor(math.log10(abs(snn_modularity)))) + 2)} & {si_format(time_snn_actual, precision=2)}s  \\\\ \n')
        f.close()
dt_end = datetime.now()
with open(f'{my_parent_path}/information.txt', 'a') as f:
        f.write(f'\n End Date and Time: {dt_end}  \\\\ \n NEW Bias Current Values \n  Edge Count: {edge_spike_count}  HeatMap Resolution: 40x40')
        f.close()

# If the graph is large and ideal partitions will take a long time to calculate, run them after SNN partitions
# if N >= 25:
#     get_ideal_partitions()