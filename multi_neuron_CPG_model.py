"""
This script is an example of how to use the MulitNeuronSimulation and Visualization_2D classes contained in LIF_model. 
It creates a simple network of two neurons which act as a central pattern generator. The neuron which is stimulated constantly,
fires a burst of three action potentials before pausing for about 25 ms. The pause is due to the second neuron acting as an
inhibitory interneuron. 

"""


import numpy as np
import matplotlib.pyplot as plt
from LIF_model import Neuron,MultiNeuronSimulation,Visualization_2D



if __name__ == '__main__':

    # membrane are in cm^2: around 2000 micro-meter^2 (for a sperical neuron with 20 micrometer radius) in cm^2 
    membrane_area = 4000e-8

    # membrane capacitance is 1 micro Farad per square centimeter 
    Cm = 1e-6 * membrane_area

    # resistance is 100 to 300 Mega Ohms 
    Rm = 300e+6

    # equilibrium potential of leak -90 mv
    El = -90e-3

    # create neurons
    neuron1 = Neuron(Rm,Cm,El)
    neuron2 = Neuron(Rm,Cm*2,El)

   # the inhibitory and excitatory connection strengths
    inhibitory_constr = -4.7e-10
    excitatory_constr= 3.05e-10

    # create a matrix stating how they are connected
    connectivity_matrix =  np.array([[0,inhibitory_constr],[excitatory_constr,0]])

    # initialize simulation object
    mult1 = MultiNeuronSimulation([neuron1,neuron2],
                                  connectivity_mat=connectivity_matrix)
    
    # function with which we stimulate the neurons
    stim_func = lambda x: 3.9e-10 if 0.001 < x < 0.3 else 0

    # set the stimulus function for the neuons
    mult1.set_all_current_functions([stim_func,None])

    # run the expeiment 
    mult1.run_experiment(stop_time_input= 0.1,step_size_input=0.002)

    # initialize a visualization of simulation
    vis = Visualization_2D(mult1,
                           list_of_centroid_coordinates=[(0,-25),(0,25)],
                           )
    
    # show the simulation
    vis.run_2D_visualization(pause=0.3)


