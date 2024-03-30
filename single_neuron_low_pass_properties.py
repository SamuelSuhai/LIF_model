"""
This script shows how to use the classes SingleNeuronSimuation and Animation. 
As an example the low pass properties of a neuron are demonstrated by creating two identical LIF neurons 
and injecting sin current of different frequencies. Low frequency current leads to action potentials,
while higher frequency current does not. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from LIF_model import Neuron,SingleNeuronSimulation,Animation


if __name__== '__main__':

    # # membrane are in cm^2: 2000 micro-meter^2 (for a sperical neuron with 20 micrometer radius) in cm^2 
    membrane_area = 4000e-8

    # membrane capacitance is 1 micro Farad per square centimeter 
    Cm = 1e-6 * membrane_area

    # resistance is 100 to 300 Mega Ohms 
    Rm = 300e+6

    # equilibrium potential of leak -90 mv
    El = -90e-3
    
    # define two identical neurons object with our parameters
    neuron1 = Neuron(Rm,Cm,El)
    neuron2 = Neuron(Rm,Cm,El)

    # create simulation objects in which we implement high frequency or low frequency stimulation
    high_freq_sim = SingleNeuronSimulation(neuron1)
    low_freq_sim = SingleNeuronSimulation(neuron2)  

    # we set the properties of the two types of stimulation (in Herz and Ampere)
    higher_frequency = 40
    lower_ferquency = 2 
    amplitude = 1.2e-10
    offset = 1.3e-10

    # define two stimulus functions: current injeccted as function of time 
    high_freq_stimulus = lambda t: offset + amplitude* np.sin(2 * np.pi * higher_frequency * t) if 0.05 < t < 1.05 else 0
    low_freq_stimulus = lambda t:  offset + amplitude* np.sin(2 * np.pi * lower_ferquency * t) if 0.05 < t < 1.05 else 0

    # include the defined current function to our simulation
    high_freq_sim.set_current_function(high_freq_stimulus)
    low_freq_sim.set_current_function(low_freq_stimulus)


    # set the simulation time length to 2s and the time resolution to 1 ms
    end_time = 1
    step_size = 0.001

    # run simulation
    high_freq_sim.run_experiment(end_time,step_size)
    low_freq_sim.run_experiment(end_time,step_size)


    # show results of low frquency stimulus simulation 
    animation_high_freq = Animation(high_freq_sim, title = f"{higher_frequency} Hz sinosoidal current. Save or close to coninue")
    animation_high_freq.run_animation()

    # wait and close previous animatino 
    plt.pause(0.5)
    plt.close('all')

    # show results of low frquency stimulus simulation 
    animation_low_freq = Animation(low_freq_sim,title = f"{lower_ferquency} Hz sinosoidal current. Save or close to coninue")
    animation_low_freq.run_animation()
    
    # wait and close previous animatino 
    plt.pause(0.5)
    plt.close('all')
