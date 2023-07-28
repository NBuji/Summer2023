#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:30:38 2023

@author: sadler

log plot showing time scaling of different partition methods

python3 adler/492/code/log_plots.py
"""

import numpy as np
from matplotlib import pyplot as plt
import os
from snn_functions import *

# Define the number of nodes in the graph as an array
num_nodes = np.array([9, 9, 13, 16, 17, 17, 18, 23, 23, 24, 26, 27, 29, 32, 33, 36, 37, 38, 41, 41, 44, 50, 50, 52, 58, 63, 68, 72, 77, 81, 88, 99, 100])

# Define the computational times for each partition method as arrays
metis_time = np.array([767.95*10**-6, 780.11*10**-6, 1.57*10**-3, 1.07*10**-3, 1.28*10**-3, 922.92*10**-6, 1.19*10**-3, 1.49*10**-3, 1.21*10**-3, 1.54*10**-3, 15.52*10**-3, 1.37*10**-3, 2.59*10**-3, 2.9*10**-3, 1.41*10**-3, 1.43*10**-3, 1.49*10**-3, 2.85*10**-3, 3.52*10**-3, 21.06*10**-3, 1.83*10**-3, 3.26*10**-3, 2.25*10**-3, 2*10**-3, 3.42*10**-3, 2.77*10**-3, 5.97*10**-3, 3.61*10**-3, 29.99*10**-3, 2.78*10**-3, 2.95*10**-3, 35.12*10**-3, 3.56*10**-3])
kernighan_lin_time = np.array([584.6*10**-6, 484.94*10**-6, 512.6*10**-6, 1.86*10**-3, 929.12*10**-6, 656.84*10**-6, 809.67*10**-6, 2.05*10**-3, 2.1*10**-3, 1.11*10**-3, 1.14*10**-3, 1.22*10**-3, 1.99*10**-3, 1.75*10**-3, 2.43*10**-3, 1.32*10**-3, 21.23*10**-3, 2.38*10**-3, 3.69*10**-3, 1.89*10**-3, 1.96*10**-3, 3.14*10**-3, 15.22*10**-3, 2.17*10**-3, 18.22*10**-3, 2.58*10**-3, 30.3610**-3, 24.86*10**-3, 19.31*10**-3, 8.33*10**-3, 81.59*10**-3, 165.67*10**-3, 3.91*10**-3])
girvan_newman_time = np.array([4.9*10**-3, 1.73*10**-3, 1.92*10**-3, 19.7*10**-3, 18.99*10**-3, 7.12*10**-3, 10.57*10**-3, 61.95*10**-3, 25.28*10**-3, 82.25*10**-3, 73.73*10**-3, 78.5*10**-3, 74.77*10**-3, 418.33*10**-3, 186.16*10**-3, 180.91*10**-3, 176.49*10**-3, 323.76*10**-3, 1.17, 345.64*10**-3, 397.43*10**-3, 107.2*10**-3, 575.14*10**-3, 567.94*10**-3, 10.86, 457.38*10**-3, 53.94, 2.78, 63.27, 41.66, 7.34, 184.8, 19.09])
SNN_time = np.array([get_simulation_time(nodes)*0.6*10**-12 for nodes in num_nodes])


log_num_nodes = np.log(num_nodes)
log_metis_time = np.log(metis_time)
log_kernighan_lin_time = np.log(kernighan_lin_time)
log_girvan_newman_time = np.log(girvan_newman_time)
log_SNN_time = np.log(SNN_time)

# Plot the log-log scatter graph
plt.scatter(log_num_nodes, log_metis_time, label='metis')
plt.scatter(log_num_nodes, log_kernighan_lin_time, label='kernighan_lin')
plt.scatter(log_num_nodes, log_girvan_newman_time, label='girvan_newman')
plt.scatter(log_num_nodes, log_SNN_time, label='SNN')

# Add a legend, labels, and title to the plot
plt.legend()
plt.xlabel('Number of Nodes (log)')
plt.ylabel('Computational Time (seconds) (log)')
plt.title('Comparison of Partition Methods')

# Add a line of best fit for the partition methods
fit_metis = np.polyfit(log_num_nodes, log_metis_time, 1)
plt.plot(log_num_nodes, fit_metis[0]*log_num_nodes + fit_metis[1], '--', color='blue')
fit_kernighan_lin = np.polyfit(log_num_nodes, log_kernighan_lin_time, 1)
plt.plot(log_num_nodes, fit_kernighan_lin[0]*log_num_nodes + fit_kernighan_lin[1], '--', color='orange')
fit_girvan_newman = np.polyfit(log_num_nodes, log_girvan_newman_time, 1)
plt.plot(log_num_nodes, fit_girvan_newman[0]*log_num_nodes + fit_girvan_newman[1], '--', color='green')
fit_SNN = np.polyfit(log_num_nodes, log_SNN_time, 1)
plt.plot(log_num_nodes, fit_SNN[0]*log_num_nodes + fit_SNN[1], '--', color='red')

# Display the equation for the line of best fit
plt.text(2.2, -11, f'metis: ln(t) = {fit_metis[0]:.3g}ln(N)', color='blue')
plt.text(2.2, -13, f'kernighan_lin: ln(t) = {fit_kernighan_lin[0]:.3g}ln(N)', color='orange')
plt.text(2.2, -15, f'girvan_newman: ln(t) = {fit_girvan_newman[0]:.3g}ln(N)', color='green')
plt.text(2.2, -17, f'SNN: ln(t) = {fit_SNN[0]:.3g}ln(N)', color='red')

# Save plot
if not os.path.exists(f'adler/492/figures/'):
        os.makedirs(f'adler/492/figures/')
plt.savefig(f'adler/492/figures/computational_time_scaling.png')
plt.clf()
plt.close()