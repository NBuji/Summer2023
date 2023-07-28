#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 6 12:35:13 2023

@author: nbujiashvili & sadler

function with the SNN architecture to solve the graph partitioning probelm for a given negative Q matrix
"""

from inspect import GEN_SUSPENDED
from brian2 import *
import numpy as np
from snn_constants_equations import *
from jj_plot_functions import *
import math

def get_SNN_partition(scaled_negQ, simulation_time, bias_currents, spike_heights, threshold_voltages, my_path, graph_seed, sigma_factor, time_factor, stoch_factor):

    start_scope()

    # Set the seed
    import random
    random_seed = random.randint(0, 2**32 - 1)

    with open(f'{my_path}/information.txt', 'a') as f:
        f.write(f'seed is: {random_seed} \n')
        f.write(f'graph seed is: {graph_seed} \n')
        f.write(f'time factor: {time_factor} \n')
        f.write(f'sigma factor is: {sigma_factor} \n')
        f.close()
    seed(random_seed)

    N = scaled_negQ.shape[0] # number of nodes/neurons

    # Create neuron groups
    neuron_equation_1 = get_neuron_equations(stoch_factor)
    G_w = NeuronGroup(N, neuron_equation_1, threshold='vp > v_th', refractory=refractory_period, method='euler')
    G_w.t_spike = 0
    G_w.i_in = 0
    G_w.i_b = bias_currents
    G_w.v_th = threshold_voltages
    # Spontaneous neurons
    spontaneous_neuron_eqs = get_spontaneous_neuron_equation(simulation_time, sigma_factor, stoch_factor) # get spontaneous neuron equation which is dependent on the simulation time due to scaling spontaneity annealing
    G_s = NeuronGroup(N, spontaneous_neuron_eqs, threshold='vp > v_th', refractory=refractory_period, method='euler')
    G_s.t_spike = 0
    G_s.i_in = 0
    G_s.i_b = bias_currents
    G_s.sig = spike_heights
    G_s.sigma = spike_heights
    G_s.v_th = threshold_voltages

    # Working neuron synapses
    S_w_w = Synapses(G_w, G_w, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
    S_w_w.connect()
    S_w_s = Synapses(G_w, G_s, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
    S_w_s.connect()
    # Spontaneous neuron synapses
    S_s_w = Synapses(G_s, G_w, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
    S_s_w.connect()
    S_s_s = Synapses(G_s, G_s, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
    S_s_s.connect()

    # Assign weights given by the negative q matrix
    for i in np.arange(N):
        for j in np.arange(N):
            weight = scaled_negQ[i][j]
            S_w_w.w[i, j] = weight
            S_w_s.w[i, j] = weight
            
            S_s_w.w[i, j] = weight
            S_s_s.w[i, j] = weight

    # State monitors
    if N < 10: # only if the graph is small otherwise this takes up too much memory
        M_w = StateMonitor(G_w, ['vp', 'i_in'], record=True)
    #M_s = StateMonitor(G_s, ['vp', 'i_in'], record=True)

    # Spike monitors
    SM_w = SpikeMonitor(G_w)
    SM_s = SpikeMonitor(G_s)
    # sigma_monitor = StateMonitor(G_s, 'sigma', record=True)
    # phi_monitor = StateMonitor(G_s, 'phi_p', record=True)


    # Spike height and time arrays to keep track of overlapping spikes
    sl_w = []
    tl_w = []
    sl_s = []
    tl_s = []
    for n in np.arange(N):
        sl_w.append([])
        tl_w.append([])
        sl_s.append([])
        tl_s.append([])

    # checks if a neuron in a neuron group has reached the threshold,
    # if so, sends out a spiking pulse,
    # updates the spiking arrays to keep track of overlapping arrays
    def check_spike(group, spike_list, time_list):
        for n in np.arange(N):
            time_list[n] = [x - 1 for x in time_list[n]] # each spike has one less second in it
            i = 0
            while i < len(time_list[n]):
                if time_list[n][i] == 0: # remove puleses which are at the end of their width
                    group.i_in[n] -= spike_list[n][i]
                    del spike_list[n][i]
                    del time_list[n][i]
                else:
                    i += 1
            if group.t_spike[n] != 0: # add spike to input current
                spike_list[n].append(group.t_spike[n])
                time_list[n].append(pulse_width)
                group.i_in[n] += group.t_spike[n]
                group.t_spike[n] = 0
            if 'group.sig[n]' in locals(): # if it has an additional spontineity let's make sure it's not actually above 0
                if (group.i_in[n] + group.sig[n]/2) < -0.1*spike_heights[n]: # if current goes too low, bring it back up
                    spike_list[n].append(-0.1*spike_heights[n] - (group.i_in[n] + group.sig[n]/2))
                    time_list[n].append(1)
                    group.i_in[n] += (-0.1*spike_heights[n] - (group.i_in[n] + group.sig[n]/2))
            else:
                if (group.i_in[n]) < -0.1*spike_heights[n]: # if current goes too low, bring it back up
                    spike_list[n].append(-0.1*spike_heights[n] - (group.i_in[n]))
                    time_list[n].append(1)
                    group.i_in[n] += (-0.1*spike_heights[n] - (group.i_in[n]))
        return spike_list, time_list

    # Arrays to store spikes and whirls for each neuron
    num_sim_intervals = math.ceil(simulation_time/500)
    if num_sim_intervals > 20:
        num_sim_intervals = 20
    spike_arrays = np.zeros((2*num_sim_intervals, num_groups, N))
    whirl_arrays = np.zeros((2*num_sim_intervals, num_groups, N))
    
    # Functions to count number of spikes and whirls for each neuron
    def count_spikes(spike_array):
        for i, SM in enumerate([SM_w, SM_s]):
            spike_trains = SM.spike_trains()
            for n in np.arange(N):
                spike_array[i][n] = len(spike_trains[n])
    def count_whirls(whirl_array):
            for i, G in enumerate([G_w, G_s]):
                for n in np.arange(N):
                    whirl_array[i][n] = round(G.phi_p[n]/(2*pi*second) - 0.2)

    # run simulation for 1 second at a time
    for sec in np.arange(simulation_time):
        run(1*second)

        # check spikes
        sl_w, tl_w = check_spike(G_w, sl_w, tl_w)
        sl_s, tl_s = check_spike(G_s, sl_s, tl_s)

        # count spikes and whirls
        counter = 0
        while counter < (num_sim_intervals):
            if ((sec + 1) == round((counter*(simulation_time - 200)/(num_sim_intervals - 1)))):
                count_spikes(spike_arrays[2*counter])
                count_whirls(whirl_arrays[2*counter])
                counter = num_sim_intervals
            elif ((sec + 1) == round((counter*(simulation_time - 200)/(num_sim_intervals - 1)) + 200)):
                count_spikes(spike_arrays[2*counter + 1])
                count_whirls(whirl_arrays[2*counter + 1])
                counter = num_sim_intervals
                if ((sec + 1) == round(((counter + 1)*(simulation_time - 200)/(num_sim_intervals - 1)))):
                    count_spikes(spike_arrays[2*(counter + 1)])
                    count_whirls(whirl_arrays[2*(counter + 1)])
            counter += 1
       
    if not os.path.exists(f'{my_path}/csv'):
        os.makedirs(f'{my_path}/csv')
    np.savetxt(f'{my_path}/csv/spike_array.csv', spike_arrays[-1] - spike_arrays[-2], delimiter=',')
    np.savetxt(f'{my_path}/csv/whirl_array.csv', whirl_arrays[-1] - whirl_arrays[-2], delimiter=',')
    

    
    plot_arrays(spike_arrays, "spikes", my_path)
    plot_array((spike_arrays[-1] - whirl_arrays[-1]), "final_spikes_whirls", f'{my_path}/spikes')
    # plot_sigma(sigma_monitor,f'{my_path}/spikes', 'sigma_graph' )
    # plot_phi(phi_monitor,f'{my_path}/spikes', 'phi_graph' )

    if N < 10:
        for n in np.arange(N):
            plot_vp(M_w, n, my_path)
            plot_i_in(M_w, n, my_path)

    return spike_arrays