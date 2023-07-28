#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sadler

Functions for dealing with heatmaps and such
"""

import seaborn as sns
from inspect import GEN_SUSPENDED
from matplotlib import pyplot as plt
from brian2 import *
import os
import numpy as np

# Create heatmap
def plot_array(array, title, my_path, name=''):
    if (name != '') and (not os.path.exists(f'{my_path}/{title}')):
        os.makedirs(f'{my_path}/{title}')
    plt.figure()
    sns.heatmap(array, annot = True, linewidth=0.5)
    plt.title(f"{name}")
    if (name != ''):
        plt.savefig(f'{my_path}/{title}/{name}.png')
    else:
        plt.savefig(f'{my_path}/{title}.png')
    plt.clf()
    plt.close()
    return

# Split array into intervals for heatmaps
def plot_arrays(arrays, title, my_path):
    for interval in np.arange(int(arrays.shape[0]/2)):
        DF = arrays[2*interval + 1] - arrays[2*interval]
        plot_array(DF, title, my_path, f'interval_{interval}')
    return

def plot_sigma(sigma_monitor, my_path, name= ''):
    plt.plot(sigma_monitor.t, sigma_monitor.sigma[0])
    plt.xlabel('Time')
    plt.ylabel('Sigma')
    plt.title('Sigma vs. Time')
    plt.show()
    title = 'sigma_vs_time.png'
    if (name != ''):
        plt.savefig(f'{my_path}/{name}.png')
    else:
        plt.savefig(f'{my_path}/{title}.png')
    plt.clf()
    plt.close()
    return

def plot_phi(phi_monitor, my_path, name= ''):
    plt.plot(phi_monitor.t, phi_monitor.phi_p[0])
    plt.xlabel('Time')
    plt.ylabel('phi')
    plt.title('phi vs. Time')
    plt.show()
    title = 'phi_vs_time.png'
    if (name != ''):
        plt.savefig(f'{my_path}/{name}.png')
    else:
        plt.savefig(f'{my_path}/{title}.png')
    plt.clf()
    plt.close()
    return

# Plot v_p vs t for a neuron
def plot_vp(output_monitor, neuron_num, my_path):
    plt.figure()
    plt.plot(output_monitor.t/second, output_monitor.vp[neuron_num])
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.title("vp JJ #%i" % neuron_num)
    if not os.path.exists(f'{my_path}/voltage'):
        os.makedirs(f'{my_path}/voltage')
    plt.savefig(f'{my_path}/voltage/{neuron_num}.png')
    plt.clf()
    plt.close()
    return

# Plot i_in vs t for a neuron
def plot_i_in(output_monitor, neuron_num, my_path):
    plt.figure()
    plt.plot(output_monitor.t/second, output_monitor.i_in[neuron_num])
    plt.xlabel("Time")
    plt.ylabel("Current")
    plt.title("i_in JJ #%i" % neuron_num)
    if not os.path.exists(f'{my_path}/current'):
        os.makedirs(f'{my_path}/current')
    plt.savefig(f'{my_path}/current/{neuron_num}.png')
    plt.clf()
    plt.close()
    return