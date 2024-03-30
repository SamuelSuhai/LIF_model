# LIF_model
The scripts provided here were part of a python programming course at University of Tuebingen. 
The LIF_model.py script contains all the classes needed to create simulations of single Leaky Integrate and Fire (LIF) neurons or groups of neurons. For more information on LIF neuron models see https://bernstein-network.de/wp-content/uploads/2021/02/03_Lecture-03-Leaky-integrate-nd-fire-model.pdf. 
The repository has two example scripts: 
1) single_neuron_low_pass_properties.py shows how to use the create a single LIF neuron simulation using the classes from LIF_model.py. The script also shows that neurons act as a low-pass filter.
2) multi_neuron_CPG_model.py demonstrates how to use the classes from LIF_model.py to simulate networks of neurons. The script runs a simple example by creating a Central Pattern Generator which fires three action potentials before being silenced temporarily by an inhibitory interneuron.
