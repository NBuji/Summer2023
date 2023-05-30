#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 00:23:21 2022

@author: sadler

Not sure which threshold to use
5_node problem

Q = np.array([[-2, 0, 0, 1, 1], [0, -2, 0, 1, 1], [0, 0, -1, 0, 1], [1, 1, 0, -2, 0], [1, 1, 1, 0, -3]])

"""
import sys
print("Path is the following")
print(sys.path)
import pandas as pd
import seaborn as sns
from inspect import GEN_SUSPENDED
from brian2 import *
from matplotlib import pyplot as plt
import numpy as np

start_scope()

N = 5 #number of JJ neurons in each group
groups = 3 #to represent the working, spontaneous, and output neuron groups
pi = np.pi
gamma = 1
small_lambda = 0.25
lambda_s = 0.5
lambda_p = 0.5
eta = 1
sig = 0.55 #stochasticity
delay_duration = 10*second
simulation_time = 1000 #how long the simulation needs to run for

neuron_eqs = '''
dphi_p/dt = vp : second
dvp/dt = - gamma*vp/second - sin(phi_p/second)/second - small_lambda*(phi_p+phi_c)/(second**2) + lambda_s*i_in/second + (1 - lambda_p)*i_b/second : 1
dphi_c/dt = vc : second
dvc/dt = - gamma*vc/second - sin(phi_c/second)/second - small_lambda/eta*(phi_c+phi_p)/(second**2) + lambda_s/eta*i_in/second - lambda_p/eta*i_b/second : 1
i_b : 1
i_in : 1
t_spike : 1
'''

stochastic_neuron_eqs = '''
dphi_p/dt = vp : second
dvp/dt = - gamma*vp/second - sin(phi_p/second)/second - small_lambda*(phi_p+phi_c)/(second**2) + lambda_s*(i_in + sig + sigma)/second + (1 - lambda_p)*i_b/second : 1
dsigma/dt = -sig/(simulation_time*second) : 1
dphi_c/dt = vc : second
dvc/dt = - gamma*vc/second - sin(phi_c/second)/second - small_lambda/eta*(phi_c+phi_p)/(second**2) + lambda_s/eta*(i_in + sig + sigma)/second - lambda_p/eta*i_b/second : 1
i_b : 1
i_in : 1
t_spike : 1
'''

Q = np.array([[-2, 0, 0, 1, 1], [0, -2, 0, 1, 1], [0, 0, -1, 0, 1], [1, 1, 0, -2, 0], [1, 1, 1, 0, -3]])

#weights
def scale_by(q_matrix, num_neurons):
    q_iis = np.zeros(num_neurons)
    for i in np.arange(num_neurons):
        q_iis[i] = abs(q_matrix[i][i])
    max = np.amax(q_iis)
    scale_factor = 1/(1.0*max)
    return scale_factor

scale_factor = scale_by(Q, N)

print("Scale Factor: ")
print(scale_factor)

Q = Q*(-scale_factor) #scale and flip

def get_bias_current(q_matrix, num_neurons):
    if not(num_neurons == len(q_matrix) == len(q_matrix[0])):
        print("ERROR: neurons and Q matrix size are not the same")
        return
    
    thresholds = np.zeros(num_neurons)
    Q_ji = np.zeros((num_neurons,num_neurons))
    for i in np.arange(num_neurons):
        for j in np.arange(num_neurons):
            Q_ji[j][i] = q_matrix[i][j]

    current_thresholds = np.zeros(N)

    for i in np.arange(num_neurons):
        max = np.amax(Q_ji[i])
        min = np.amin(Q_ji[i])
        thresholds[i] = (max + min)
        current_thresholds[i] = max

    #Move thresolds above 1
    threshold_min = np.amin(thresholds)
    for n in np.arange(num_neurons):
        thresholds[n] += (1 - threshold_min)

    bias_current_values = np.zeros(num_neurons)
    for n in np.arange(num_neurons):
        bias_current_values[n] = 1.9*((thresholds[n])**(-1/16)) #not sure how to scale this part
    
    return bias_current_values, current_thresholds

bias_current, th_currents = get_bias_current(Q, N)

print("Bias Current: ")
print(bias_current)
print("Th Current: ")
print(th_currents)

Q = Q*3


print(Q)

spiking_rate = 0.01*Hz
P = PoissonGroup(3, rates = spiking_rate)

#Create groups
G_w = NeuronGroup(N, neuron_eqs, threshold='vp > 1', refractory=10*second, method='euler')
G_w.t_spike = 0
G_w.i_in = 0
G_w.i_b = bias_current
G_s = NeuronGroup(N, stochastic_neuron_eqs, threshold='vp > 1', refractory=10*second, method='euler')
G_s.t_spike = 0
G_s.i_in = 0
G_s.i_b = bias_current
G_o = NeuronGroup(N, neuron_eqs, threshold='vp > 1', refractory=10*second, method='euler')
G_o.t_spike = 0
G_o.i_in = 0
G_o.i_b = bias_current

#Input Poisson spike synapses
S_P_w = Synapses(P, G_w, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_P_w.connect()
S_P_s = Synapses(P, G_s, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_P_s.connect()
S_P_o = Synapses(P, G_o, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_P_o.connect()

#Working neuron synapses
S_w_w = Synapses(G_w, G_w, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_w_w.connect()
S_w_s = Synapses(G_w, G_s, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_w_s.connect()
S_w_o = Synapses(G_w, G_o, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_w_o.connect()

#Spontaneous neuron synapses
S_s_w = Synapses(G_s, G_w, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_s_w.connect()
S_s_s = Synapses(G_s, G_s, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_s_s.connect()
S_s_o = Synapses(G_s, G_o, 'w : 1', on_pre='t_spike_post += w', delay=delay_duration)
S_s_o.connect()

for i in np.arange(N):
    for j in np.arange(N):
        weight = scale_factor*Q[i][j]
        S_P_w.w[i, j] = weight
        S_P_s.w[i, j] = weight
        S_P_o.w[i, j] = weight

        S_w_w.w[i, j] = weight
        S_w_s.w[i, j] = weight
        S_w_o.w[i, j] = weight
        
        S_s_w.w[i, j] = weight
        S_s_s.w[i, j] = weight
        S_s_o.w[i, j] = weight

print("Q after second loop")
print(Q)
#State monitors
M_w = StateMonitor(G_w, ['vp', 'i_in', 'phi_p'], record=True)
M_s = StateMonitor(G_s, ['vp', 'i_in', 'phi_p'], record=True)
M_o = StateMonitor(G_o, ['vp', 'i_in', 'phi_p'], record=True)

SM_w = SpikeMonitor(G_w)
SM_s = SpikeMonitor(G_s)
SM_o = SpikeMonitor(G_o)

pulse_width = 6 #width of spikes

#spike height and time arrays to keep track of overlapping spikes
sl_w = []
tl_w = []
sl_s = []
tl_s = []
sl_o = []
tl_o = []
for n in np.arange(N):
    sl_w.append([])
    tl_w.append([])
    sl_s.append([])
    tl_s.append([])
    sl_o.append([])
    tl_o.append([])

#Checks if a neuron in a neuron group has reached the threshold,
#and if so, sends out a spikeing pulse
#then updates the spiking arrays to keep track of overlapping arrays
def check_spike(group, spike_list, time_list):
    for n in np.arange(N):
        time_list[n] = [x - 1 for x in time_list[n]]
        count = 0
        for i in np.arange(len(time_list[n])):
            if time_list[n][i - count] == 0:
                group.i_in[n] -= spike_list[n][i - count]
                del spike_list[n][i - count]
                del time_list[n][i - count]
                count += 1
        if group.t_spike[n] != 0:
            spike_list[n].append(group.t_spike[n])
            time_list[n].append(pulse_width)
            group.i_in[n] += group.t_spike[n]
            group.t_spike[n] = 0
        if group.i_in[n] < -0.3*th_currents[n]:
            spike_list[n].append((-0.3*th_currents[n] - group.i_in[n]))
            time_list[n].append(1)
            group.i_in[n] += (-0.3*th_currents[n] - group.i_in[n])
    return spike_list, time_list

spikes_200 = np.zeros((groups,N))
spikes_400 = np.zeros((groups,N))
spikes_600 = np.zeros((groups,N))
spikes_800 = np.zeros((groups,N))
spikes_1000 = np.zeros((groups,N))

def count_spikes(spike_array):
    for i, SM in enumerate([SM_w, SM_s, SM_o]):
        spike_trains = SM.spike_trains()
        for n in np.arange(N):
            spike_array[i][n] = len(spike_trains[n])

whirls_200 = np.zeros((groups,N))
whirls_400 = np.zeros((groups,N))
whirls_600 = np.zeros((groups,N))
whirls_800 = np.zeros((groups,N))
whirls_1000 = np.zeros((groups,N))

def count_whirls(whirl_array):
    for i, G in enumerate([G_w, G_s, G_o]):
        for n in np.arange(N):
            whirl_array[i][n] = round(G.phi_p[n]/(2*pi*second))

#Checks spikes in groups every second
for sec in np.arange(simulation_time):
    run(1*second)
    sl_w, tl_w = check_spike(G_w, sl_w, tl_w)
    sl_s, tl_s = check_spike(G_s, sl_s, tl_s)
    sl_o, tl_o = check_spike(G_o, sl_o, tl_o)

    if (sec + 1) == 200:
        count_spikes(spikes_200)
        count_whirls(whirls_200)
    elif (sec + 1) == 400:
        count_spikes(spikes_400)
        count_whirls(whirls_400)
    elif (sec + 1) == 600:
        count_spikes(spikes_600)
        count_whirls(whirls_600)
    elif (sec + 1) == 800:
        count_spikes(spikes_800)
        count_whirls(whirls_800)
    elif (sec + 1) == 1000:
        count_spikes(spikes_1000)
        count_whirls(whirls_1000)


#Plot figures for each neuron group
fig, axs = plt.subplots(ncols = 3)
fig.suptitle("Heatmaps of Spikes")

data_1 = np.zeros((groups,N))
data_2 = np.zeros((groups,N))
data_3 = np.zeros((groups,N))

for n in np.arange(N):
    for gr in np.arange(groups):
        data_1[gr][n] = spikes_200[gr][n]
        data_2[gr][n] = spikes_600[gr][n] - spikes_400[gr][n]
        data_3[gr][n] = spikes_1000[gr][n] - spikes_800[gr][n]

# plotting the heatmap
df1 = pd.DataFrame(data_1)
df2 = pd.DataFrame(data_2)
df3 = pd.DataFrame(data_3)

sns.heatmap(df1, ax = axs[0], annot = True, linewidth=0.5)
sns.heatmap(df2, ax = axs[1], annot = True, linewidth=0.5)
sns.heatmap(df3, ax = axs[2], annot = True, linewidth=0.5)

axs[0].set_title("0 - 200s")
axs[1].set_title("400 - 600s")
axs[2].set_title("800 - 1000s")

# displaying the plotted heatmap
plt.show()
plt.savefig('figures/5_node/spikes_heatmap.png')

#Plot whirls for each neuron group
fig, axs = plt.subplots(ncols = 3)
fig.suptitle("Heatmaps of whirls")

whirl_data_1 = np.zeros((groups,N))
whirl_data_2 = np.zeros((groups,N))
whirl_data_3 = np.zeros((groups,N))

for n in np.arange(N):
    for gr in np.arange(groups):
        whirl_data_1[gr][n] = whirls_200[gr][n]
        whirl_data_2[gr][n] = whirls_600[gr][n] - whirls_400[gr][n]
        whirl_data_3[gr][n] = whirls_1000[gr][n] - whirls_800[gr][n]

# plotting the heatmap
wdf1 = pd.DataFrame(whirl_data_1)
wdf2 = pd.DataFrame(whirl_data_2)
wdf3 = pd.DataFrame(whirl_data_3)

sns.heatmap(wdf1, ax = axs[0], annot = True, linewidth=0.5)
sns.heatmap(wdf2, ax = axs[1], annot = True, linewidth=0.5)
sns.heatmap(wdf3, ax = axs[2], annot = True, linewidth=0.5)

axs[0].set_title("00 - 200s")
axs[1].set_title("400 - 600s")
axs[2].set_title("800 - 1000s")

# displaying the plotted heatmap
plt.show()
plt.savefig('figures/5_node/whirl_heatmap.png')

#Plot difference for each neuron group
fig, axs = plt.subplots(ncols = 3)
fig.suptitle("Whirls - Spikes")

sns.heatmap(wdf1 - df1, ax = axs[0], annot = True, linewidth=0.5)
sns.heatmap(wdf2 - df2, ax = axs[1], annot = True, linewidth=0.5)
sns.heatmap(wdf3 - df3, ax = axs[2], annot = True, linewidth=0.5)

axs[0].set_title("00 - 200s")
axs[1].set_title("400 - 600s")
axs[2].set_title("800 - 1000s")

# displaying the plotted heatmap
plt.show()
plt.savefig('figures/5_node/difference_heatmap.png')

fig, axs = plt.subplots(2)
fig.suptitle("Watching Output Spikes")
axs[0].plot(M_o.t/second, M_o.vp[1])
axs[0].set_xlabel("Time")
axs[0].set_ylabel("vp JJ #%i" % 1)
axs[1].plot(M_o.t/second, M_o.vp[2])
axs[1].set_xlabel("Time")
axs[1].set_ylabel("vp JJ #%i" % 2)
plt.show()
plt.savefig('figures/5_node/spikes.png')

fig, axs = plt.subplots(3)
fig.suptitle("Watching Output Spikes")
axs[0].plot(M_o.t/second, M_o.vp[0])
axs[0].set_xlabel("Time")
axs[0].set_ylabel("vp JJ #%i" % 0)
axs[1].plot(M_o.t/second, M_o.vp[1])
axs[1].set_xlabel("Time")
axs[1].set_ylabel("vp JJ #%i" % 1)
axs[2].plot(M_o.t/second, M_o.vp[2])
axs[2].set_xlabel("Time")
axs[2].set_ylabel("vp JJ #%i" % 2)
plt.show()
plt.savefig('figures/5_node/spikes_0-2.png')

fig, axs = plt.subplots(3)
fig.suptitle("Watching i_in")
axs[0].plot(M_o.t/second, M_o.i_in[0])
axs[0].set_xlabel("Time")
axs[0].set_ylabel("i_in JJ #%i" % 0)
axs[1].plot(M_o.t/second, M_o.i_in[1])
axs[1].set_xlabel("Time")
axs[1].set_ylabel("i_in JJ #%i" % 1)
axs[2].plot(M_o.t/second, M_o.i_in[2])
axs[2].set_xlabel("Time")
axs[2].set_ylabel("i_in JJ #%i" % 2)
plt.show()
plt.savefig('figures/5_node/i_in_0-2.png')

fig, axs = plt.subplots(2)
fig.suptitle("Watching Output Spikes")
axs[0].plot(M_o.t/second, M_o.vp[3])
axs[0].set_xlabel("Time")
axs[0].set_ylabel("vp JJ #%i" % 3)
axs[1].plot(M_o.t/second, M_o.vp[4])
axs[1].set_xlabel("Time")
axs[1].set_ylabel("vp JJ #%i" % 4)
plt.show()
plt.savefig('figures/5_node/spikes_3-4.png')

fig, axs = plt.subplots(2)
fig.suptitle("Watching i_in")
axs[0].plot(M_o.t/second, M_o.i_in[3])
axs[0].set_xlabel("Time")
axs[0].set_ylabel("i_in JJ #%i" % 3)
axs[1].plot(M_o.t/second, M_o.i_in[4])
axs[1].set_xlabel("Time")
axs[1].set_ylabel("i_in JJ #%i" % 4)
plt.show()
plt.savefig('figures/5_node/i_in_3-4.png')
