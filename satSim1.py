import matplotlib.pyplot as plt
import satParam as P
import numpy as np
from signalGenerator import signalGenerator
from satAnimation import satAnimation
from dataPlotter import dataPlotter
from satDynamics import satDynamics
from matplotlib import animation
from satControl import satControl

# instantiate pendulum, controller, and reference classes
sat = satDynamics(alpha=0.9)
reference = signalGenerator(amplitude=0.5, frequency=0.02)   
force = signalGenerator(amplitude=1, frequency=1)
rel_states=satControl()
# instantiate the simulation plots and animation
dataPlot = dataPlotter()
anim = satAnimation() 
u=np.array([0,0,0, 0 ,0, 0, 0, 0, 0])
t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        r = reference.square(t)
        #u = force.sin(t)
        
        y = sat.update(u)  # Propagate the dynamics
        Z = rel_states.satRelMotion(sat.state, u)
        #print(Z)

        u=np.array([0,0,0, 0, 0, 0, float(Z[6]) ,float(Z[7]), float(Z[8])])
        t = t + P.Ts  # advance time by Ts
        #print('r=',y[13])
        #print(y[0])
    # update animation and data plots at rate t_plot
    anim.update(sat.state)
    #ani=animation.FuncAnimation(anim.fig, anim.update(sat.state),frames=P.frames, interval=40, blit=True)
    
    print(Z[3])
    dataPlot.update(t, r, sat.state,Z, 0)
    plt.pause(0.00001)  # allows time for animation to draw
#plt.show()

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
