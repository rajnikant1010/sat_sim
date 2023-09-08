from http.client import USE_PROXY
import numpy as np 
from math import cos, sin 
import satParam as P


class satControl:
    def __init__(self):
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

   # def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input force
        #u = saturate(u, self.force_limit)
    #    self.rk4_step(u, self.f)  # propagate the state by one time sample
    #    y = self.h()  # return the corresponding output
    #    return y
    
    #def satOrbDynamics(self, state, u):


     #   pdot = np.sqrt(p/self.k)*2*p/q*u_theta
     #   fdot = np.sqrt(p/self.k)*(np.sin(L)*u_r+1/q*((q+1)*np.cos(L)+f)*u_theta+f/q*(np.sin(L)-k*np.cos(L))*u_h)
     #   gdot = np.sqrt(p/self.k)*(-np.cos(L)*u_r+1/q((q+1)*np.sin(L)+g)*u_theta+f/q*(h*np.sin(L)-k*np.cos(L))*u_h)
     #   hdot = np.sqrt n c(p/self.k)*s**2*np.cos(L)/(2*q)*u_h
     #   kdot = np.sqrt(p/self.k)*s**2*np.sin(L)/(2*q)*u_h
     #   Ldot = np.sqrt(p/self.k)*(h*np.sin(L)-k*np.cos(L)*u_h+np.sqrt(self.k*p)*(q/p)**2)
        
    def satRelMotion(self, state, u):
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

        r_d = state[13][0]
        rdot_d = state[14][0]
        theta_d = state[15][0]
        thetadot_d = state[16][0]
        phi_d = state[17][0]
        phidot_d = state[18][0]
        

        x=r*np.cos(phi)*np.cos(theta)
        y=r*np.cos(phi)*np.sin(theta)
        z=r*np.sin(phi)
        xdot=rdot*np.cos(phi)*np.cos(theta)-r*np.sin(phi)*np.cos(theta)*phidot-r*np.cos(phi)*np.sin(theta)*thetadot
        ydot=rdot*np.cos(phi)*np.sin(theta)-r*np.sin(phi)*np.sin(theta)*phidot+r*np.cos(phi)*np.cos(theta)*thetadot
        zdot=rdot*np.sin(phi)+r*np.cos(phi)*phidot


        x_d=r_d*np.cos(phi_d)*np.cos(theta_d)
        y_d=r_d*np.cos(phi_d)*np.sin(theta_d)
        z_d=r_d*np.sin(phi_d)
        xdot_d=rdot_d*np.cos(phi_d)*np.cos(theta_d)-r_d*np.sin(phi_d)*np.cos(theta_d)*phidot_d-r_d*np.cos(phi_d)*np.sin(theta_d)*thetadot_d
        ydot_d=rdot_d*np.cos(phi_d)*np.sin(theta_d)-r_d*np.sin(phi_d)*np.sin(theta_d)*phidot_d+r_d*np.cos(phi_d)*np.cos(theta_d)*thetadot_d
        zdot_d=rdot_d*np.sin(phi_d)+r_d*np.cos(phi_d)*phidot_d
        
        Eul=self.quat2Eul313(state)

        ROT=self.Rotation_inertial2body(theta, phi, 0)

        W= np.array([[0, -thetadot, 0],
                     [thetadot, 0, -phidot],
                     [-0, phidot, 0]])
        rho=np.matmul(ROT,(np.array([[x_d], [y_d], [z_d]])-np.array([[x], [y], [z]])))
        rho_dot=np.matmul(ROT,(np.array([[xdot_d], [ydot_d], [zdot_d]])-np.array([[xdot], [ydot], [zdot]])))-np.matmul(W,np.matmul(ROT,(np.array([[x_d], [y_d], [z_d]])-np.array([[x], [y], [z]]))))


        #################



        #####

        #r_ddot=r*thetadot**2*(cos(phi))**2+r*phidot**2-(self.k/r**2)
        #theta_ddot=((-2*rdot*thetadot)/r)-((2*thetadot*phidot*sin(phi))/cos(phi))
        #phi_ddot=-thetadot**2*cos(phi)*sin(phi)-((2*rdot*phidot)/r)


        ####

        #r_ddotd=r_d*thetadot_d**2*(cos(phi_d))**2+r_d*phidot_d**2-(self.k/r_d**2)+(u[6]/self.m)
        #theta_ddotd=((-2*rdot_d*thetadot_d)/r_d)-((2*thetadot_d*phidot_d*sin(phi_d))/cos(phi_d))+(u[7]/(self.m*r_d*cos(phi_d)))
        #phi_ddotd=-thetadot_d**2*cos(phi_d)*sin(phi_d)-((2*rdot_d*phidot_d)/r_d)+(u[8]/(self.m*r_d))

        F=np.array([[thetadot**2*np.cos(phi)**2 +phidot**2+2*self.k/(r**2),   0,        -r*thetadot**2*2*np.cos(phi)*np.sin(phi)],
                    [2*rdot*thetadot/(r**2),                                  0,         2*thetadot*phidot/(cos(phi)**2)],
                    [2*rdot*phidot/(r**2),                                    0,         -thetadot**2*(1-2*sin(phi)**2)]])
        

        G=np.array([[0,               2*r*thetadot*cos(phi)**2,                2*r*phidot],
                    [-2*thetadot/r,  -2*rdot/r - 2*phidot*sin(phi)/cos(phi),  -2*thetadot*sin(phi)/cos(phi)],
                    [-2*phidot/r,    -2*thetadot*cos(phi)*sin(phi),            -2*rdot/r]])
        ####
        r_r=r_d-r
        theta_r=theta_d-theta
        phi_r=phi_d-phi


        r_rdot=rdot_d-rdot
        theta_rdot= thetadot_d-thetadot
        phi_rdot=phidot_d-phidot

        #########
        #gain Matrices

        K1=300*np.array([[1.5,0.9, 0.8],
                     [0.9, 1.2, 0.7],
                     [0.8, 0.7, 0.95]])
        
        K2=np.array([[75,20, 50],
                     [20, 75, 50],
                     [50, 50, 200]])

        U=  -np.matmul((F+K1),np.array([[r_r], [theta_r], [phi_r]]))  -np.matmul((K1),np.array([[r_rdot], [theta_rdot], [phi_rdot]]))
        #print(U)    
        #returns relative states and control input for deputy satellite.
        Z=np.array([[rho[0]],[rho_dot[0]], [rho[1]], [rho_dot[1]], [rho[2]], [rho_dot[2]], [U[0]], [U[1]], [U[2]]])


        return Z
    


    def Rotation_inertial2body(self,theta, phi, psi):
        R1=np.array([[np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [0,            0           , 1]])
        
        R2=np.array([[1,            0           , 0],
                     [0, np.cos(phi), np.sin(phi)],
                     [0, -np.sin(phi), np.cos(phi)]])
        
        R3=np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0,            0           , 1]])
        ROT=np.matmul(R3,np.matmul(R2,R1))
        return ROT

    def quat2Eul313(self, state):
        q1=state[9][0]
        q2=state[10][0]
        q3=state[11][0]
        q4=state[12][0]
        R=np.array([[q4**2+q1**2-q2**2-q3**2, 2*(q1*q2-q4*q3), 2*(q4*q2+q1*q3)],
                    [2*(q1*q2+q4*q3), q4**2-q1**2+q2**2-q3**2, 2*(q2*q3-q1*q4)],
                    [2*(q1*q3-q4*q2), 2*(q4*q1+q2*q3), q4**2-q1**2-q2**2+q3**2]])
        psi=np.arctan2(R[0][2],R[1][2])
        theta=np.arccos(R[2][2])
        phi=np.arctan2(R[2][0],-R[2][1])
        y=np.array([[psi],[theta], [phi]])
        return y

    def rk4_step(self, u, f):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = f(self.state, u)
        F2 = f(self.state + self.Ts / 2 * F1, u)
        F3 = f(self.state + self.Ts / 2 * F2, u)
        F4 = f(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)