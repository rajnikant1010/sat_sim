import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

plt.ion()  # enable interactive drawing


class dataPlotter:
    ''' 
        This class plots the time histories for the pendulum data.
    '''

    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 3    # Number of subplot rows
        self.num_cols = 2    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        #self.zref_history = []  # reference position z_r
        self.r_history = []  # radial distance r
        self.theta_history = []  # angle theta
        self.rdot_history = []  # control force
        self.phi_history = [] 
        self.phidot_history = [] 
        self.thetadot_history = [] 
        self.w1_history = [] 
        self.w2_history = []
        self.w3_history = []
        self.q1_history = []
        self.q2_history = []
        self.q3_history = []
        self.q4_history = []


        self.rd_history = []  # radial distance r
        self.thetad_history = []  # angle theta
        self.rdotd_history = []  # control force
        self.phid_history = [] 
        self.phidotd_history = [] 
        self.thetadotd_history = [] 


        # Relative States
        self.xrel_history = []
        self.xdotrel_history = []
        self.yrel_history = []
        self.ydotrel_history = []
        self.zrel_history = []
        self.zdotrel_history = []
        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0][0], ylabel='r(m)', title='Satellite'))
        self.handle.append(myPlot(self.ax[0][1], ylabel='rdot(m)'))
        self.handle.append(myPlot(self.ax[1][0], ylabel='theta(rad)'))
        self.handle.append(myPlot(self.ax[1][1], ylabel='thetadot(rad)'))

        self.handle.append(myPlot(self.ax[2][0], xlabel='t(s)', ylabel='phi(rad)'))
        self.handle.append(myPlot(self.ax[2][1], xlabel='t(s)', ylabel='phi(rad)'))

        ########################
        self.fig1, self.ax1 = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        self.handle1 = []
        self.handle1.append(myPlot(self.ax1[0][0], ylabel='x relative(m)', title='Satellite relative parameters'))
        self.handle1.append(myPlot(self.ax1[0][1], ylabel='xdot relative(m)'))
        self.handle1.append(myPlot(self.ax1[1][0], ylabel='y relative'))
        self.handle1.append(myPlot(self.ax1[1][1], ylabel='ydot relative'))

        self.handle1.append(myPlot(self.ax1[2][0], xlabel='t(s)', ylabel='z relative'))
        self.handle1.append(myPlot(self.ax1[2][1], xlabel='t(s)', ylabel='zdot relative'))



    def update(self, t, reference, states, states_rel, ctrl):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # update the time history of all plot variables
        self.time_history.append(t)  # time
        #self.zref_history.append(reference)  # reference base position
        self.r_history.append(states[0,0])  # base position
        self.theta_history.append(states[2,0])  # rod angle (converted to degrees)
        self.phi_history.append(states[4,0])  # force on the base
        self.rdot_history.append(states[1,0]) 
        self.thetadot_history.append(states[3,0]) 
        self.phidot_history.append(states[5,0]) 
        self.w1_history.append(states[6,0]) 
        self.w2_history.append(states[7,0]) 
        self.w3_history.append(states[8,0]) 
        self.q1_history.append(states[9,0]) 
        self.q2_history.append(states[10,0]) 
        self.q3_history.append(states[11,0]) 
        self.q4_history.append(states[12,0]) 

        self.rd_history.append(states[13,0])  # base position
        self.thetad_history.append(states[15,0])  # rod angle (converted to degrees)
        self.phid_history.append(states[17,0])  # force on the base
        self.rdotd_history.append(states[14,0]) 
        self.thetadotd_history.append(states[16,0]) 
        self.phidotd_history.append(states[18,0]) 


        # Append relative states

        self.xrel_history.append(states_rel[0,0])
        self.xdotrel_history.append(states_rel[1,0])
        self.yrel_history.append(states_rel[2,0])
        self.ydotrel_history.append(states_rel[3,0])
        self.zrel_history.append(states_rel[4,0])
        self.zdotrel_history.append(states_rel[5,0])


        # update the plots with associated histories
        self.handle[0].update(self.time_history, [self.rd_history])
        self.handle[1].update(self.time_history, [self.rdotd_history])
        self.handle[2].update(self.time_history, [self.thetad_history])
        self.handle[3].update(self.time_history, [self.thetadotd_history])
        self.handle[4].update(self.time_history, [self.phid_history ])
        self.handle[5].update(self.time_history, [self.phidotd_history ])

        # update the plots
        self.handle1[0].update(self.time_history, [self.xrel_history])
        self.handle1[1].update(self.time_history, [self.xdotrel_history])
        self.handle1[2].update(self.time_history, [self.yrel_history])
        self.handle1[3].update(self.time_history, [self.ydotrel_history])
        self.handle1[4].update(self.time_history, [self.zrel_history ])
        self.handle1[5].update(self.time_history, [self.zdotrel_history ])


class myPlot:
    ''' 
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True   

    def update(self, time, data):
        ''' 
            Adds data to the plot.  
            time is a list, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
           

