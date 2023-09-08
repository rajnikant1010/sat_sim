from http.client import USE_PROXY
import numpy as np 
from math import cos, sin
import satParam as P

class satDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions # r, rdot, theta, thetadot, phi, phidot
        self.state = np.array([
            [P.r0],  # z initial position
            [P.rdot0],  # Theta initial orientation
            [P.theta0],  # zdot initial velocity
            [P.thetadot0],  # Thetadot initial velocity
            [P.phi0],
            [P.phidot0],
            [P.w10],
            [P.w20],
            [P.w30],
            [P.q10],
            [P.q20],
            [P.q30],
            [P.q40],
            [P.r_d0],
            [P.rdot_d0],
            [P.theta_d0],
            [P.thetadot_d0],
            [P.phi_d0],
            [P.phidot_d0],
        ])
        # simulation time step
        self.Ts = P.Ts
        self.k=P.k
        self.m=P.m
        self.force_limit = P.F_max
        self.I1=P.I1
        self.I2=P.I2
        self.I3=P.I3

    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input force
        #u = saturate(u, self.force_limit)
        self.rk4_step(u, self.f)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output
        return y

    def f(self, state, u):
        # Return xdot = f(x,u)
        r = state[0][0]
        rdot = state[1][0]
        theta = state[2][0]
        thetadot = state[3][0]
        phi = state[4][0]
        phidot = state[5][0]
        w1=self.state[6][0]
        w2=self.state[7][0]
        w3=self.state[8][0]
        q1=self.state[9][0]
        q2=self.state[10][0]
        q3=self.state[11][0]
        q4=self.state[12][0]
        
        
        # The equations of motion.
        r_ddot=r*thetadot**2*(cos(phi))**2+r*phidot**2-(self.k/r**2)+(u[0]/self.m)
        theta_ddot=((-2*rdot*thetadot)/r)-((2*thetadot*phidot*sin(phi))/cos(phi))+(u[1]/(self.m*r*cos(phi)))
        phi_ddot=-thetadot**2*cos(phi)*sin(phi)-((2*rdot*phidot)/r)+(u[2]/(self.m*r))

        w1_dot=(((self.I2-self.I3)/(self.I1))*w2*w3)+u[3]/self.I1
        w2_dot=(((self.I3-self.I1)/self.I2)*w3*w1)+u[4]/self.I2
        w3_dot=(((self.I1-self.I2)/self.I3)*w1*w2)+u[5]/self.I3

        q_dot=0.5*np.matmul(np.array([[0, w3, -w2, w1],
                              [-w3, 0, w1, w2],
                              [w2, -w1, 0, w3],
                              [-w1, -w2, -w3, 0]]),np.array([q1,
                                                             q2,
                                                             q3, 
                                                             q4]))
        ######
        #defining relative dynamics
        #thetaddot_d1=-2*rdot/r*thetadot_d1
        #xd_ddot=x_d*(thetadot_d1**2+2*self.k/(r**3))+y_d*thetaddot_d1+2*yd_dot*thetadot_d1+u[6]
        r_d = state[13][0]
        rdot_d = state[14][0]
        theta_d = state[15][0]
        thetadot_d = state[16][0]
        phi_d = state[17][0]
        phidot_d = state[18][0]
        
        
        
        # The equations of motion.
        r_ddotd=r_d*thetadot_d**2*(cos(phi_d))**2+r_d*phidot_d**2-(self.k/r_d**2)+(u[6]/self.m)
        theta_ddotd=((-2*rdot_d*thetadot_d)/r_d)-((2*thetadot_d*phidot_d*sin(phi_d))/cos(phi_d))+(u[7]/(self.m*r_d*cos(phi_d)))
        phi_ddotd=-thetadot_d**2*cos(phi_d)*sin(phi_d)-((2*rdot_d*phidot_d)/r_d)+(u[8]/(self.m*r_d))
        #yd_ddot=x_d*thetaddot_d1-2*xd_dot*thetadot_d1+y_d*(thetadot_d1**2-self.k/(r**3))+u[7]

        #zd_ddot=-z_d*self.k/(r**3)+u[8]
        # build xdot and return
        xdot = np.array([[rdot], [r_ddot], [thetadot], [theta_ddot], [phidot], [phi_ddot], [w1_dot], [w2_dot], [w3_dot], [q_dot[0]], [q_dot[1]], [q_dot[2]], [q_dot[3]], [rdot_d], [r_ddotd], [thetadot_d], [theta_ddotd], [phidot_d], [phi_ddotd]])
        return xdot
    
    #def Orbital_dynamics(self, state, state_O, u):
    #    r = state[0][0]
    #    rdot = state[1][0]
    #    theta = state[2][0]
    #    thetadot = state[3][0]
    #    phi = state[4][0]
    #    phidot = state[5][0]
    #
    #   a=state_O[0][0]
    #    e=state_O[1][0]
    #    i=state_O[2][0]
    #    Omega=state_O[3][0]
    #    omega=state_O[4][0]
    #    M=state_O[5][0]
    #
    #    f=theta-omega
    #    p=a*(1-e)
    #    h=thetadot*r**2
    #    n=mu/a**3
    #    neta=np.sqrt(1-(e*np.cos(omega))**2-(e*np.sin(omega))**2)
        # Osculating orbit 
    #    a_dot=2*a**2/h*(e*np.sin(f)*u_r+p/r*u_theta)
    #    e_dot=1/h*(p*np.sin(f)*u_r+((p+r)*np.cos(f)+r*e)*u_theta)
    #    i_dot=r*np.cos(theta)/h*u_h
    #    Omega_dot=r*np.sin(theta)/(h*np.sin(i))*u_h
    #    omega_dot=1/(h*e)*(-p*np.cos(f)*u_r+(p+r)*np.sin(f)*u_theta)-r*np.sin(theta)*np.cos(theta)/(h*np.sin(i))*u_h
    #    M_dot=n+neta/(h*e)*((p*np.cos(f)-2*r*e)*u_r-(p+r)*np.sin(f)*u_theta) 



    def h(self):
        # return y = h(x)
        r = self.state[0][0]
        rdot = self.state[1][0]
        theta = self.state[2][0]
        thetadot = self.state[3][0]
        phi = self.state[4][0]
        phidot = self.state[5][0]
        w1=self.state[6][0]
        w2=self.state[7][0]
        w3=self.state[8][0]
        q1=self.state[9][0]
        q2=self.state[10][0]
        q3=self.state[11][0]
        q4=self.state[12][0]
        r_d = self.state[13][0]
        rdot_d = self.state[14][0]
        theta_d = self.state[15][0]
        thetadot_d = self.state[16][0]
        phi_d = self.state[17][0]
        phidot_d = self.state[18][0]

        y = np.array([[r],[rdot], [theta], [thetadot], [phi], [phidot],[w1], [w2], [w3], [q1], [q2], [q3], [q4], [r_d], [rdot_d], [theta_d], [thetadot_d], [phi_d], [phidot_d]])
        return y

    def rk4_step(self, u, f):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = f(self.state, u)
        F2 = f(self.state + self.Ts / 2 * F1, u)
        F3 = f(self.state + self.Ts / 2 * F2, u)
        F4 = f(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

        
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
    return u
