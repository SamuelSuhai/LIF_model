# import modules TO DO: check if all necessary 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import types




class Neuron():

    def __init__(self,Rm: float, Cm: float,El,threshold: float = -0.03, name: str = "") -> None:
        """
        Initialize a Neuron object.

        parameters:
        - Rm: Membrane resistance is Ohms.
        - Cm: Membrane capacitance in Farads.
        - El: Reversal potential of leak in Volt.
        - threshold: Threshold for spiking. Default is -0.03 Volt.
        - name: Name of the neuron. Default is an empty string.
        """
        
        # membrane resistance
        self.Rm = Rm

        # membrane capacitance
        self.Cm = Cm

        # reversal potential of leak
        self.El = El

        # time constant of membrane 
        self.tau = self.Rm * self.Cm

        # set initial membrane potential to reversal potential of leak current
        self.Vm = El

        # threshold for spiking 
        self.thresh = threshold

        # max voltage of Action potential
        self.Vm_spike = threshold + 0.07

        # set true when neuron is spiking
        self.spiking = False

        # set the neurons name
        self.name = name

    def update_membrane_potential(self,stimulus: float,time_step_length: float) -> bool:
        """ Update the membrane potential at time t as a function of previous membrane potentials. 
        
        Parameters:
        - stimulus: The stimulus, i.e. current, applied to the neuron in Ampere.
        - time_step_length: The time step length used in the simulation in Seconds.

        Returns:
        - bool: True if a spike just occurred, False if not.
        """
        

        # calculate current equilibrium potential given the injected stimulus
        updated_equilibrium_potential = self.El + self.Rm * stimulus


        # update membrane potential according to current
        self.Vm =  updated_equilibrium_potential + (self.Vm - updated_equilibrium_potential) * np.exp(-time_step_length/self.tau)


        # check if membane potential over threhold 
        if self.Vm >= self.thresh:
            
            # record that the neuron is spiking
            self.spiking  = True
            
            # set membrane potential to resting membrane potential 
            self.Vm = self.El   
        
        else:
            # record neuron is not spiking
            self.spiking = False



class SingleNeuronSimulation():
    """
    Class for simulating the behavior of a single neuron.

    Attributes:
        neuron_object: The neuron object to simulate.
        stimulus_history: History of stimulation.
        Vm_history: History of membrane potential of the neuron.
        spike_history: History of spiking activity.
        time_stamps: Time stamps in the simulation.
        step_size: Step size used in the simulation.

    Methods:
        set_current_function: Allows user to define current injected by time into the neuron.
        get_current_trace: Applies the current function to simulation time stamps.
        run_experiment: run the expeiment and save the reults.


    """
    
    def __init__(self,neuron_object: Neuron) -> None:
        """
        Initialize SingleNeuronSimulation object.

        Parameters:
            neuron_object (type: Neuron): The neuron object used in the simulation.
            
        """
        
        # Store neuron 
        self.neuron_object = neuron_object

        # history of stimulaiton
        self.stimulus_history = None

        # history of membrane potential of neuron
        self.Vm_history = None
        
        # history of spiking
        self.spike_history = None

        # time stamps in simulation
        self.time_stamps = None

        self.step_size = None

    def set_current_function(self,current_function):
        """
        Allows user to set the function defining how much current is injected at a certain time into the neuorn.

        Parameters:
            current_function (function): A function of time (float).
        """
        
        # check if we defined a function
        if isinstance(current_function,types.FunctionType):            

            # store the function that defines the sitmulus
            self.current_fucntion = current_function

        else:
            raise Exception("Current function was not identified as a function. Use for example lambda functions.")

    def get_current_trace(self,stop_time: float,step_size: float):
        """
        Calculate the current trace based on the specified stop time and step size.

        Parameters:
            stop_time: The stop time of the simulation.
            step_size: The step size used in the simulation.
        """
        
        # set attributes of current
        self.stop_time = stop_time 
        self.step_size = step_size

        # create array with stimulus time stamps 
        self.time_stamps = np.arange(0,stop_time,step_size)
        self.time_stamp_nr = len(self.time_stamps)

        # create array with current injected (in Ampere) per timestep
        self.stimulus_history = np.zeros(shape=(len(self.time_stamps)))
        
        # here we define the current trace as the stimulus history
        for idx,time_step in enumerate(self.time_stamps):
            self.stimulus_history[idx] = self.current_fucntion(time_step)



    def run_experiment(self,stop_time_input: float,step_size_input: float = 0.01):
        """
        Run the neuron simulation experiment.

        Parameters:
            stop_time_input: The stop time of the simulation.
            step_size_input: The step size used in the simulation. Defaults to 0.01.
        """

        # let user know the simulation process started 
        print("Performing single neuron simulation ... ")
        
        # define a current trace
        self.get_current_trace(stop_time = stop_time_input,step_size= step_size_input)

        # initialize arrays for storing membrane potential  
        self.Vm_history = np.zeros(shape=(len(self.time_stamps)))
        self.spike_history = np.zeros(shape=(len(self.time_stamps)))

        # loop over time stamps  
        for time_stamp_idx in range(self.time_stamp_nr):
            
            # index the current injected in the time step now
            stimulation_now = self.stimulus_history[time_stamp_idx]
            
            # update the neurons membrane potential 
            self.neuron_object.update_membrane_potential(stimulation_now,self.step_size)


            # record results
            self.Vm_history[time_stamp_idx] = self.neuron_object.Vm
            self.spike_history[time_stamp_idx] = self.neuron_object.spiking


        

class Animation():
    """
    A class to do matplotlib animations based on single neuron simulations.

    Attributes:
        simulation_object: An object of SingleNeuronSimulation representing the neuron simulation data.
        x_data: List of time stamps of membrane potental.
        y_data: The membrane potential to plot.
        x_data_current: Time stamps of the current injected.
        y_data_current: The current injected per time stamp.
        title: The title for the animation.
        fig,axes: The figure/axes objects for the animation.
        line: The line artist for membrane potential plot.
        current_line: The line artist for the current trace plot.
    
    Methods:
        
    """

    def __init__(self,simulation_object: SingleNeuronSimulation,title : str):
        """
        Initialize Animation with a SingleNeuronSimulation object and a title.

        Parameters:
            simulation_object: An object of SingleNeuronSimulation representing the neuron simulation data.
            title: The title for the animation.
        """

        
        # store simulation object
        self.simulation_object = simulation_object
        
        # lists to store data for x and y membrane potential trace in 2D Animation 
        self.x_data = []
        self.y_data = []

        # lists to store current trace data for plotting 2D animation
        self.x_data_current,self.y_data_current = [],[]

        # title 
        self.title = title

    def set_up_plot(self,y_axis_buffer: float = 1.2):
        """
        Set up the plot for the animation.

        Parameters:
            y_axis_buffer: Buffer for y-axis limits in the plot. The factor by which the limits are multiplied by.
        """
        
        # set up a figure with 2 axes subplots
        self.fig, self.axes = plt.subplots(2)

        # line object on the axes for the membane potential
        self.line, = self.axes[0].plot([], [], lw=2)

        # line object for the current trace
        self.current_line, = self.axes[1].plot([],[],lw=2)

        # get the maximum Vm for plotting purposes
        max_Vm =  self.simulation_object.neuron_object.Vm_spike
        min_Vm = np.min(self.simulation_object.Vm_history)

        # set up limits of membrane potential plot
        self.axes[0].set_xlim(0, self.simulation_object.stop_time)
        self.axes[0].set_ylim(min_Vm,max_Vm)
        
        # set up limits for current trace plot 
        self.axes[1].set_xlim(0, self.simulation_object.stop_time)
        min_y = -np.max (self.simulation_object.stimulus_history)
        max_y = -np.min (self.simulation_object.stimulus_history)
        self.axes[1].set_ylim(min_y - y_axis_buffer*(max_y-min_y),
                              max_y + y_axis_buffer*(max_y - min_y))

        # add title 
        self.fig.suptitle(self.title)

        # add labels 
        self.axes[0].set_xlabel('Time (mS)')
        self.axes[0].set_ylabel('Membrane Potential (Volt)')
        self.axes[1].set_xlabel('Time (mS)')
        self.axes[1].set_ylabel('Current Injected (Ampere)')

        # make some space between the subplots
        plt.tight_layout()

    def include_spike_in_plotting(self,frame,fraction_of_step):
        """ 
        Function that includes additional x and y data if there was a spike.
        It plots the membrane potential first at threshold then at spike level.
        This is necessary since LIF models do not model the shape of the spike.
        There the spike is just a thin line.

        Prameters:
            fraction_of_steps: The fraction of the time step size unsed to plot the Vm at threhold and spike level.
        
        """
        
        # plot the membrane potential just at threshold
        self.x_data.append(self.simulation_object.time_stamps[frame] - fraction_of_step*self.simulation_object.step_size)
        self.y_data.append(self.simulation_object.neuron_object.thresh)

        # plot the spike just before return to resting membrane potential at spike peak 
        self.x_data.append(self.simulation_object.time_stamps[frame] - fraction_of_step/2*self.simulation_object.step_size)
        self.y_data.append(self.simulation_object.neuron_object.Vm_spike)

    def animation_update_func(self,frame,fraction_of_step = 0.1):
        """
        Function to update the lines in animation.

        Parameters:
            fraction_of_step: This argument is passed to the include_spikes_in_plotting function. 

        Note:
            By convention positive current injected into the cell is plotted with an inverted y-axis.

        """
        
  

        # check if spike occured
        if self.simulation_object.spike_history[frame]:

            # add some data to visualize the spike
            self.include_spike_in_plotting(frame,fraction_of_step)
                

        # otherwise just plot the membrane potential
        self.x_data.append(self.simulation_object.time_stamps[frame])
        self.y_data.append(self.simulation_object.Vm_history[frame])
    
        # append injected current to list, to plot it. 
        self.x_data_current.append(self.simulation_object.time_stamps[frame])

        # Note: we invert the sign by convetion
        self.y_data_current.append( - self.simulation_object.stimulus_history[frame])

        # set the membrane potential data to line object
        self.line.set_data(self.x_data, self.y_data)

        # set the data for the injected current. 
        self.current_line.set_data(self.x_data_current,self.y_data_current)

        return self.line,self.current_line    
    
    def run_animation(self):
        """
        Method that runs the animation.
        """


        # set up a plot 
        self.set_up_plot()
        
        # create animation
        ani = animation.FuncAnimation(self.fig, self.animation_update_func,
                                      frames = self.simulation_object.time_stamp_nr,
                                      interval=2, blit=True, repeat=False)

        # show animation
        plt.show()




#######################################


class MultiNeuronSimulation():
    """
    A class to simulate multiple LIF neurons with synaptic connections.

    Parameters:
        neuron_list: A list of Neuron objects representing the neurons in the simulation.
        connectivity_mat: A numpy 2d-matrix repsresenting innervation patterns between neurons.

    Attributes:
        neuron_list: A list of Neuron objects representing the neurons in the simulation.
        stimulated_neurons_idx: A list of indices of neurons that receive stimulation via a pipette.
        all_current_functions: A list storing the current functions for each stimulated neuron.
        connectivity_mat: A 2d-matrix representing connections. From column inxed to row index.
        stimulus_history: A 2d-matrix representing the history of stimulation.
        Vm_history: A 2d-matrix representing the history of membrane potentials of all neurons.
        spike_history: A 2d-matrix representing the history (rows) of spiking for different neurons (columns).
        time_stamps: A np.array representing the time stamps in the simulation.
        step_size: The time step size in the simulation.
        synaptic_delay: The delay between presynaptic action potential and postsynaptic current.
        synaptic_input: A 2d-Matrix representing the synaptic input to each neuron (columns) at different times (rows).
        PSC_duration: The duration of postsynaptic current.
    """
    
    def __init__(self, neuron_list: list,connectivity_mat: np.ndarray) -> None:
        """
        Initialize MultiNeuronSimulation with a list of neurons and a connectivity matrix.

        Parameters:
            neuron_list: A list of Neuron objects representing the neurons in the simulation.
            connectivity_mat: A numpy.ndarray representing the connectivity matrix between neurons.
                              Matrix must be square, have zero on diagonal entries, and the number of columns 
                              must correspond to the number of neurons. The column index indicates which neuron 
                              the axon arises froma and the row index indicates the target. The strength of the 
                              connection is the entry of the matrix (current in Ampere injected into the target).
                              Example: np.array([[0,3e-12,0],
                                                 [0,0,0],
                                                 [0,0,0]]). The second neuron in the self.neuron_list innervated the
                              first neuron in the list. When the second one is stimulated the first one receivers 
                              3 pico Ampere current for the duration of self.PSC_duration.
                              

        """
        
        # Store all neurons 
        self.neuron_list = neuron_list

        # List of indices in self.neuron_list of neurons theat we inject current in
        self.stimulated_neurons_idx = []

        # store the functions used to stimulate the neurons here
        self.all_current_functions = [None for n in range(len(neuron_list))]

        # store connectivity_mat if in correct form
        self.check_and_store_conmat(connectivity_mat)

        # history of stimulaiton
        self.stimulus_history = None

        # history of membrane potentials of all neurons
        self.Vm_history = None
        
        # history of spiking
        self.spike_history = None

        # time stamps in simulation
        self.time_stamps = None

        self.step_size = None

        self.synaptic_delay = 0.001

        self.synaptic_input = None

        self.PSC_duration = 0.010

    def check_and_store_conmat(self,conmat: np.ndarray):
        """
        Check the validity of the connectivity matrix and store it.
        """

        # make sure neurons are not connected with eachother
        if any(np.diag(conmat)):
            raise Exception("All diagonal entries of the connectivity matrix must be zero. Neuron cannont inervate itself")
        
        # make sure the matrix is square
        elif conmat.shape[0] != conmat.shape[1]:
            raise Exception(f"Connectivity Matrix specified must be diagonal, shape given {conmat.shape}")
        
        # make sure it has correct number of shapes for neuron
        elif conmat.shape[1] != len(self.neuron_list):
            raise Exception(f"Connectivity Matrix row/col number ({conmat.shape} given) should match number of neurons ({len(self.neuron_list)})")
        else:
            # store correct connectivity matrix
            self.connectivity_mat = conmat


    def set_all_current_functions(self,current_function_list: list):
        """
        Set current functions for all neurons, if the input is correct.

        Parameters:
            Current functions should be a function of time (float) or None if a neuron should not be stimulated.
            Example:[None, lambda x: 1e-12 if x > 0.1] --> The first neuron in self.neuron_list is not stimulated. 
                    The second one is stimulated with 1 pico Ampere starting from 0.1 seconds.
            """

        # make sure we have the correct number of elements in the current function list
        if len(current_function_list) != len(self.neuron_list):
            raise Exception (f"Number of current functions specified ({len(current_function_list)}) does not match number of neurons ({len(self.neuron_list)})")
        
        # check if entries are fucntions valid or None-types
        for neuron_idx,function in enumerate(current_function_list):
            
            # check if we defined a function
            if isinstance(function,types.FunctionType):
                
                # store the vectorized function that defines the sitmulus
                # set output type to float to avoid numerical problems with small output numbers
                self.all_current_functions[neuron_idx] = np.vectorize(current_function_list[neuron_idx],otypes=[float])

                # note down that this neuron is a neuron we stimulate in the simulatino 
                self.stimulated_neurons_idx.append(neuron_idx)

            # check if the input is neither a function nor a None and raise an error
            elif function != None:
                raise Exception("An object was added to the current function list that is neither a python function nor None. All elements must be one of the two.")

        

    def get_current_trace(self,stop_time: float,step_size: float):
        """
        Generate the current trace for each neuron.

        Parameters:
            stop_time: The stop time of the simulation.
            step_size: The time step size in the simulation.
        """
        
        # set attributes of current
        self.stop_time = stop_time 
        self.step_size = step_size

        # create array with stimulus time stamps 
        self.time_stamps = np.arange(0,stop_time,step_size)
        self.time_stamp_nr = len(self.time_stamps)

        # create matrix with current injected (in Ampere) per timestep in rows and different neurons in columns
        self.stimulus_history = np.zeros(shape=(len(self.time_stamps),len(self.neuron_list)))
        

        # loop over stimulus functions 
        for neuron_idx,stim_func in enumerate(self.all_current_functions):
            
            # if we defined a function then call it on all time steps
            if stim_func is not None:
                self.stimulus_history[:,neuron_idx] = stim_func(self.time_stamps)


    def update_neurons_Vm (self,time_stamp_index: int, step_sz:float):
        """
        Update membrane potentials of neurons.

        Parameters:
            time_stamp_index: The index of the current time stamp.
            step_sz: The time step size in the simulation.
        """

        # loop over neurons and update Vm
        for neuron_idx,neuron_object in enumerate(self.neuron_list):
            
            
            total_current_now = self.synaptic_input[time_stamp_index,neuron_idx] + self.stimulus_history[time_stamp_index,neuron_idx]
            
            # update membrane potential of neurons
            neuron_object.update_membrane_potential(total_current_now,step_sz)
            
            
                    
    def store_results_of_iteration(self,time_stamp_index: int):
        # for a givven time stamp index store the data
        
         for neuron_idx,neuron_object in enumerate(self.neuron_list):
            
            # record data in this simulation step
            self.Vm_history[time_stamp_index,neuron_idx] = neuron_object.Vm
            self.spike_history[time_stamp_index,neuron_idx] = neuron_object.spiking
 
    
    def update_synaptic_input(self,time_stamp_index: int,nr_of_delay_steps:int,PSC_step_duration:float):
        
            """
            Update synaptic inputs based on spikes from the previous time steps.

            Parameters:
                time_stamp_index: The index of the current time stamp.
                nr_of_delay_steps: Number of steps between presynaptic action potential and postsynaptic current.
                PSC_step_duration: The duration of postsynaptic current in time stamps.
            """
            
            # check if there were any spikes
            if any(self.spike_history[time_stamp_index,:]):

                # make sure we are not out of bounds with the index when approaching the end of the simulation 
                if time_stamp_index + nr_of_delay_steps + PSC_step_duration <= len(self.time_stamps):

                    # start synaptic input after synaptic delay and make it last an according number of steps
                    self.synaptic_input[time_stamp_index + nr_of_delay_steps : time_stamp_index + nr_of_delay_steps + PSC_step_duration,:] += self.spike_history[time_stamp_index,:] @ self.connectivity_mat.T

                else:
                    # otherwise index the remaining entries of the matrix
                    self.synaptic_input[time_stamp_index + nr_of_delay_steps : ,:] += self.spike_history[time_stamp_index,:] @ self.connectivity_mat.T
                    


    def run_experiment(self,stop_time_input:float,step_size_input:float = 0.01):
        """
        Run the simulation experiment.

        Parameters:
            stop_time_input: The stop time of the simulation.
            step_size_input: The time step size in the simulation. Defaults to 0.01.
        """
        # let user know the simulaton has started 
        print("Running multi-neuron simulation ...")

        # define a current trace
        self.get_current_trace(stop_time = stop_time_input,step_size= step_size_input)

        # initialize arrays for storing membrane potential  
        self.Vm_history = np.zeros(shape=(len(self.time_stamps),len(self.neuron_list)))
        self.spike_history = np.zeros(shape=(len(self.time_stamps),len(self.neuron_list)))
        self.synaptic_input = np.zeros(shape=(len(self.time_stamps),len(self.neuron_list)))

        # calculate number of steps needed between presynaptic AP and post synaptic current
        delay_step_nr = int(np.max([np.round(self.synaptic_delay /self.step_size,1),1]))

        # calculate number of steps a post synaptic current acts 
        PSC_step_nr = int(np.max([np.round(self.PSC_duration  /self.step_size,1),1]))

        # loop over time stamps  
        for time_stamp_idx in range(self.time_stamp_nr):
            
            # update membrane potential
            self.update_neurons_Vm(time_stamp_idx, self.step_size)

            # record spiking and current Vms
            self.store_results_of_iteration(time_stamp_idx)
            
            # update synaptic inputs
            self.update_synaptic_input(time_stamp_idx,delay_step_nr,PSC_step_nr)

        

class Visualization_2D():
    """
    A class that visualizes a simulation of a network of LIF neurons.

    Attributes:
        simulation_object: The multi-neuron simulation object.
        list_of_centroid_coordinates: List of centroid coordinates of neurons.
        neurons: List of neuron objects to be plotted.
        stimulated_neurons_idx: Indices of stimulated neurons.
        neuron_centroids: List of neuron centroid coordinates.
        neuron_radius: Radius of neurons in the plot.
        field_size: Size of the field for visualization.
        neuron_artists: List of artists representing neurons on the plot.
        axon_artists: List of artists representing axons on the plot.
        pipette_artists: List of artists representing pipettes on the plot.
        title: Title of the plot.
        legend_position: Position of the legend on the plot.
        time_label_position: Position of the time label on the plot.
    """


    def __init__(self,simulation_object: MultiNeuronSimulation,list_of_centroid_coordinates:list) -> None:
        """
        Initialize class inctance.

        Parameters:
            simulation_object: the MulitNeuronSimulation to visualize.
            list_of_centroid_coordinates: a list containing the x and y coordinates (in that order) of where to plot the neurons. 
                                        E.g: [(0,0,),(10,10)] --> the first neuron is plottet at 0,0 second at 10,10.
        """
        

        # store mulitneuron simulation here
        self.simulation_object = simulation_object
        
        # store neuron objects to be plotted here
        self.neurons = self.simulation_object.neuron_list
    
        # store which neurons we stimulate
        self.stimulated_neurons_idx = self.simulation_object.stimulated_neurons_idx

        # check if we have as many coordinates as neurons in the simulation
        if len(list_of_centroid_coordinates) == len(self.neurons):

            # set the list of coordinates 
            self.neuron_centroids = list_of_centroid_coordinates
        else:
            raise Exception (f"Provided only {len(list_of_centroid_coordinates)} coordinates for {len(self.neurons)} neurons ...")

        # store how large neurons are
        self.neuron_radius = 10

        # store how large the field
        self.field_size = 200

        # representations of neurons on plot (artist)
        self.neuron_artists = []

        self.axon_artists = []

        self.pipette_artists = []

        self.title = "Mulit-Neuron Simulation"

        self.legend_position = [0.15,0.9]

        self.time_label_position = [0.45* self.field_size/2,0.9* self.field_size/2]

        
        


    def set_up_2D_visualization(self):

        
        # set up plot
        self.fig,self.ax = plt.subplots(figsize = (5,5))

        # set axis limits 
        self.ax.set_xlim (- self.field_size / 2, self.field_size / 2)
        self.ax.set_ylim (- self.field_size / 2, self.field_size / 2)

        
        # add title 
        self.fig.suptitle(self.title)

        # add legend
         
       



    def create_neuron_artist(self,centroid):
    
        """
        Create a neuron artist with given centroid coordinates.

        Parameters:
            centroid: Tuple with x and y coordinates of the neurons centroid.

        Returns:
            Neuron artist object.
        
        """

        # create neuron triangle object
        neuron_circle = plt.Circle(centroid, radius = self.neuron_radius, color = 'red', alpha= 0.0)

        # add to list of artists 
        self.neuron_artists.append(neuron_circle) 


        return (neuron_circle)



    def create_all_pipette_artists(self,angle_list:list,membrane_distance_factor:float = 0.9 , pipette_length = 20):
        """
        Create pipette artists for stimulated neurons.

        Parameters:
            angle_list: List of angles for pipette direction.
            membrane_distance_factor: Factor for positioning pipettes relative to neurons.
            pipette_length: Length of pipettes.
        """
        
        # loop over neurons that are stimulated and create a pipette artist
        for neuron_idx in self.stimulated_neurons_idx:


            # get start postition of pipette: close to the neuron we inject current to (i.e. the first)
            x_start = self.neuron_centroids[neuron_idx][0] + np.cos(angle_list[neuron_idx]) * self.neuron_radius * membrane_distance_factor
            y_start = self.neuron_centroids[neuron_idx][1] + np.sin(angle_list[neuron_idx]) * self.neuron_radius * membrane_distance_factor

            # the starting position is just to the left of the neuron we inject current
            x_stop = x_start + np.cos(angle_list[neuron_idx]) * pipette_length
            y_stop = y_start + np.sin(angle_list[neuron_idx]) * pipette_length

            # create pipette artist and store as in list
            self.pipette_artists.append(plt.Line2D((x_start , x_stop),(y_start ,y_stop),color = 'blue',lw = 5, alpha = 0.1))


    def update_pipette_alpha(self,time_stamp_idx:int ,max_absolute_current:float):
        """
        Update the alpha value of pipette artists based on injected current.

        Parameters:
            time_stamp_idx: Index of the current time stamp.
            max_absolute_current: Maximum absolute value of injected current.
        """
        
        # loop over pipette artists 
        for pipette_idx,pipette_artist in enumerate(self.pipette_artists):
            
            # find the pipettes corresponding neuron 
            neuron_idx = self.stimulated_neurons_idx[pipette_idx]

            # find how much current is injected into this neuron 
            current_injected = self.simulation_object.stimulus_history[time_stamp_idx,neuron_idx]

            # update pipette artists alpha value 
            alpha =  0.5 + 0.5 * current_injected / max_absolute_current
            pipette_artist.set_alpha (alpha)

    def create_axon_artist(self,from_centroid ,to_centroid,connectivity_strength, max_lw = 5, max_connectivity = 3e-10,membrane_distance_factor = 1.2,endtriangle_radius_ratio = 0.5, displacement_factor = 0 ):
        """
        Create an axon artist and an end-triangle artist representing the terminal bouton. 

        Parameters:
            from_centroid: Centroid coordinates of the neuron the axon originates from.
            to_centroid: Centroid coordinates of the neuron the axon projects to.
            connectivity_strength: Strength of connectivity between neurons.
            max_lw: Maximum line width for axon.
            max_connectivity: Maximum connectivity strength.
            membrane_distance_factor: Factor for positioning axons relative to neurons.
            endtriangle_radius_ratio: Ratio of the end triangle's height to the neuron's radius.
            displacement_factor: Factor for displacing axons to avoid overlap if two neurons innervate eachother.

        """
        
        # determine color of axon 
        if connectivity_strength < 0:
            
            # if it is inhibitory make it red
            axon_color = 'red'
        
        else:    
            
            # green if excitatory
            axon_color = 'green'

        # determine axon width
        axon_width = np.abs (max_lw * (connectivity_strength / max_connectivity))
        

        # get the cos and sin of the axon with the horizontal line
        neuron_distance =  np.sqrt((to_centroid[0] - from_centroid[0])**2 + (to_centroid[1] - from_centroid[1])**2)
        cos = (to_centroid[0] - from_centroid[0]) / neuron_distance
        sin = (to_centroid[1] - from_centroid[1]) / neuron_distance

        # calculate x, and y coordinates of lines start and stop
        x_start = from_centroid[0] + cos * self.neuron_radius * membrane_distance_factor
        x_stop = to_centroid[0] - cos * self.neuron_radius * membrane_distance_factor
        y_start = from_centroid[1] + sin * self.neuron_radius * membrane_distance_factor
        y_stop = to_centroid[1] - sin * self.neuron_radius * membrane_distance_factor

        # calculate the displacement vector, which is perpendicular to the axon 
        displacement_vector = [displacement_factor * sin, displacement_factor * -cos]


        # create line artist object starting from the membrane of the circle (not centroid)
        axon_artist = plt.Line2D((x_start + displacement_vector[0], x_stop + displacement_vector[0]),
                                 (y_start + displacement_vector[1] ,y_stop + displacement_vector[1]),
                                 color = axon_color,
                                 lw = axon_width)
        
        # Calculate the vertices of the endtrialge 
        t1 = [-self.neuron_radius* endtriangle_radius_ratio* np.sqrt(3)/2 * cos + to_centroid[0] - cos *self.neuron_radius+ displacement_vector[0],
              - self.neuron_radius * endtriangle_radius_ratio* np.sqrt(3)/2 * sin + to_centroid[1] - sin *self.neuron_radius  + displacement_vector[1]]
        
        t2 = [-sin * endtriangle_radius_ratio * self.neuron_radius/2  + to_centroid[0] - cos *self.neuron_radius+ displacement_vector[0], 
              cos * self.neuron_radius * endtriangle_radius_ratio/2 + to_centroid[1] - sin *self.neuron_radius+ displacement_vector[1]]
        
        t3 = [sin *endtriangle_radius_ratio* self.neuron_radius /2 + to_centroid[0] - cos *self.neuron_radius+ displacement_vector[0],
              -cos * endtriangle_radius_ratio*self.neuron_radius/2  + to_centroid[1] - sin *self.neuron_radius+ displacement_vector[1]]
        
        # create artist
        end_triangle = plt.Polygon((t1,t2,t3),closed = True, facecolor=axon_color)

        # store object
        self.axon_artists.append((axon_artist,end_triangle))


    def create_all_axon_artists(self):
        """
        Method to create an axon for each entry in the connectivity matrix
        
        """

        # calculate the max absolute vaulue of connectivity strength
        max_con_str = np.max(np.abs(self.simulation_object.connectivity_mat))
        
        for from_neuron_idx in range(len(self.neurons)):
            for to_neuron_idx in range(len(self.neurons)):
                connectivity_strength = self.simulation_object.connectivity_mat[to_neuron_idx,from_neuron_idx]
                
                # check if there exists a connection
                if connectivity_strength != 0: 
                    
                    # to avoid overlap between reciprocally connected neurons
                    # calculate displacement factor: 
                    if self.simulation_object.connectivity_mat[from_neuron_idx,to_neuron_idx] != 0:
                        displ_fact = self.neuron_radius / 4

                    else:
                        displ_fact = 0 

                    # add axon to plot 
                    self.create_axon_artist(self.neuron_centroids[from_neuron_idx],
                                          self.neuron_centroids[to_neuron_idx],
                                          connectivity_strength=connectivity_strength,
                                          max_connectivity=max_con_str,
                                          displacement_factor= displ_fact
                    )           

    def create_all_neuron_artists(self):
        """
        Create artists of all neurons in self.neurons
        """

        # create all neuron 
        for center in self.neuron_centroids:

            # add neuron neuron to axis
            self.create_neuron_artist(centroid=center)



    def update_neuron_alphas(self,time_stamp_idx, alpha_offset = 0.15):
        """
        Update alpha values of neuron artists based on membrane potential.

        Parameters:
            time_stamp_idx: Index of the current time stamp.
            alpha_offset: offset for alpha values.

        """
        
        # get the neurons membrane potential in the current time step
        Vms = self.simulation_object.Vm_history[time_stamp_idx,:]
        
        # get the maximum possible membrane potential: spike Vm. Assume this is equal for all neurons in the list
        spike_Vm = self.neurons[0].Vm_spike

        # get resting potential: The equilibrium potential of the leak current 
        V_rest = self.neurons[0].El 

        # calculate alphas as the fraction of maximum deviation from Vrest 
        minimum_Vm = V_rest - 0.05
        alphas =  alpha_offset + ((Vms - minimum_Vm)/ (spike_Vm  - minimum_Vm))

        # update alphas of all neurons
        for artist_idx in range(len(self.neuron_artists)):

            # check if neuron is spiking and set the highest possible alpha value to it if so
            if self.simulation_object.spike_history[time_stamp_idx,artist_idx]:
                self.neuron_artists[artist_idx].set_alpha(1)
            else:
                # make sure alphas are not smaller than zero 
                self.neuron_artists[artist_idx].set_alpha(alphas[artist_idx] if alphas[artist_idx] >= 0 else 0 )
            
        
            
    
    def draw_one_iteration(self,pause):
        """
        Draw one iteration of the visualization.

        Parameters:
            pause: Time between iterations.
        """

    

        # set limits
        self.ax.set_xlim (-self.field_size / 2, self.field_size / 2)
        self.ax.set_ylim (- self.field_size / 2, self.field_size / 2)

        # draw all neurons
        for neuron_artist in self.neuron_artists:
            self.ax.add_artist(neuron_artist)

        # loop over list where each entry contains the axon and its terminal 
        for axon_artist in self.axon_artists:
            
            # first draw axon then its terminal artist (both stored in axon_artist) 
            for artist in axon_artist:
                self.ax.add_artist(artist) 

        # draw all pipette artists 
        for artist in self.pipette_artists:
            self.ax.add_artist(artist)
            
        # draw axis and wait            
        plt.draw()
        plt.pause(pause)
        plt.cla()



    def run_2D_visualization(self, pause= 0.1):


        
        # set up visualization plot
        self.set_up_2D_visualization()

        
        # create artists
        self.create_all_neuron_artists()
        self.create_all_axon_artists()

        # for simplicity set the angles of pipettes to 180 degrees. Change this if needed.
        self.create_all_pipette_artists(angle_list= [np.pi for n in self.stimulated_neurons_idx])

        # get max and min of current injected to plot pipette alpha
        max_current_abs = np.max(np.abs(self.simulation_object.stimulus_history))
        
        # make sure we avoid later division by zero
        if max_current_abs == 0:
            max_current_abs = 1e-12

        # loop over time steps
        for time_stamp_idx in range(self.simulation_object.time_stamp_nr):

            # update alpha values to indicate membrane potential
            self.update_neuron_alphas(time_stamp_idx=time_stamp_idx)
            self.update_pipette_alpha(time_stamp_idx,max_absolute_current=max_current_abs)
            
            # draw axis wait and clear data 
            self.draw_one_iteration(pause = pause)

            # add time stamp as text artist to axis 
            self.ax.text(self.time_label_position[0],self.time_label_position[1],
                         f"Time: {np.round(self.simulation_object.time_stamps[time_stamp_idx],4)} S")

            # add a text to figure that describes what is axon, what neuron, what pipette
            self.fig.text(self.legend_position[0],self.legend_position[1],
                         "Circle: neuron (the more red the more depolarized)\n"+
                         "Blue line: pipette (the more blue the more current injected)\n"+
                         "Line and triange: axon (green is excitatory, red inhibitory)",
                         fontsize = 4)
            
        # close all figures
        plt.close(self.fig)