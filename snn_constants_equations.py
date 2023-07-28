#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nbujiashvili & sadler

JJ neuron and SNN constants and equations
"""

from brian2 import *
import numpy as np

save_to = './results_3/7_24_2023/'
pi = np.pi
num_groups = 2 # to represent the working and spontaneous neuron groups

# JJ neuron constants
gamma = 1
small_lambda = 0.25
lambda_s = 0.58
lambda_p = 0.42
eta = 1
stochasticity = 0.06 # neuron stochasticity OLD VALUE: 0.06
delay_duration = 6*second  # delay from synapse spike to neuron receiving the spike
refractory_period = 10*second # refractory period
pulse_width = 6 # width of spikes sent by synapses

# constants related to finding the ideal partition
alpha = 1
beta = 1

# for SNN functions. These numbers are now pretty arbitrary. I should work on them
min_height = 0.43
slope_change = 0.17
bc_sh_gradient = -(0.4482759 - slope_change)
bc_sh_intercept = 2.23172424 - slope_change*0.74
min_th_spike_height = 0.74

# normal JJ neuron with encoded stoachasticity
def get_neuron_equations(stoch_factor):

    neuron_eqs = f'''
    dphi_p/dt = vp : second
    dvp/dt = - {gamma}*vp/second - sin(phi_p/second)/second - small_lambda*(phi_p+phi_c)/(second**2) + lambda_s*i_in/second + (1 - lambda_p)*i_b/second + stochasticity*({stoch_factor})*xi*second**-0.5: 1
    dphi_c/dt = vc : second
    dvc/dt = - {gamma}*vc/second - sin(phi_c/second)/second - small_lambda/eta*(phi_c+phi_p)/(second**2) + lambda_s/eta*i_in/second - lambda_p/eta*i_b/second : 1
    i_b : 1
    i_in : 1
    t_spike : 1
    v_th : 1
    '''
    return neuron_eqs

def get_spontaneous_neuron_equation(simulation_time, sigma_factor, stoch_factor):
    # spontaneous neurons have added spontaneity which decays to zero over time 
    #We introduce the sigma factor to vary the time needed for the spontaneous firing to die down indepently from the simulation time
    # *((sigma + (abs(sigma))/(2*sigma + 0.0000001))    /sigma_factor  + sig
    spontaneous_neuron_eqs = f'''
    dphi_p/dt = vp : second
    dvp/dt = - {gamma}*vp/second - sin(phi_p/second)/second - small_lambda*(phi_p+phi_c)/(second**2) + lambda_s*(i_in  + sigma)/second + (1 - lambda_p)*i_b/second + stochasticity*({stoch_factor})*xi*second**-0.5: 1
    dphi_c/dt = vc : second
    dvc/dt = - {gamma}*vc/second - sin(phi_c/second)/second - small_lambda/eta*(phi_c+phi_p)/(second**2) + lambda_s/eta*(i_in + sigma)/second - lambda_p/eta*i_b/second : 1
    dsigma/dt = ((-1*sig)/({simulation_time/sigma_factor}*second))*((sigma + (abs(sigma)))/(2*sigma + 0.0000001)) : 1 
    i_b : 1
    i_in : 1
    t_spike : 1
    sig : 1
    v_th : 1
    '''
    return spontaneous_neuron_eqs

