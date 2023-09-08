# Satellite Parameter File
import numpy as np

# Physical parameters of the inverted pendulum known to the controller
G=6.674*10**(-11) # % Earth's Gravitational Constant)
k=398600.4418 #G*M (M is the mass of the earth)
m=1000 # mass of satellite in KG

I1=14.2 #Inertia diagonal elements
I2=17.3
I3=20.3
# M ? # mass of earth

# parameters for animation
w = 0.5       # Width of the cart, m
h = 0.15      # Height of the cart, m
gap = 0.005   # Gap between the cart and x-axis
radius = 0.06 # Radius of circular part of pendulum

# Initial Conditions
r0=1500+6378
omega0=np.sqrt(k/r0**3)
phi0=0*np.pi/4
theta0=0.001
rdot0=0
phidot0=0
thetadot0=omega0
w10=0
w20=0
w30=0
q10=0
q20=0
q30=0
q40=1
r_d0=(1500+6378)
rdot_d0=0
theta_d0=np.pi/20
thetadot_d0=np.sqrt(k/r_d0**3)
phi_d0=np.pi/25
phidot_d0=0

# Simulation Parameters
t_start = 0.0  # Start time of simulation
t_end = 15000.0  # End time of simulation
Ts = 1  # sample time for simulation
t_plot = 50  # the plotting and animation is updated at this rate
frames=int(t_end/Ts)
# saturation limits
F_max = 5.0                # Max Force, N

