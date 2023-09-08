from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import animation
import numpy as np 
import satParam as P
from mpl_toolkits.mplot3d import Axes3D

# if you are having difficulty with the graphics, 
# try using one of the following backends  
# See https://matplotlib.org/stable/users/explain/backends.html
# import matplotlib
# matplotlib.use('qtagg')  # requires pyqt or pyside
# matplotlib.use('ipympl')  # requires ipympl
# matplotlib.use('gtk3agg')  # requires pyGObject and pycairo
# matplotlib.use('gtk4agg')  # requires pyGObject and pycairo
# matplotlib.use('gtk3cairo')  # requires pyGObject and pycairo
# matplotlib.use('gtk4cairo')  # requires pyGObject and pycairo
# matplotlib.use('tkagg')  # requires TkInter
# matplotlib.use('wxagg')  # requires wxPython


class satAnimation:
    
    def __init__(self):
        self.flag_init = True  # Used to indicate initialization
        # Initialize a figure and axes object
        #self.fig, self.ax, self.ay, self.az = plt.subplots(111, projection='3d')
        #self.fig=plt.figure()
        #self.ax=plt.axes(projection='3d')

        # Initializes a list of objects (patches and lines)
        self.handle = []
        # Specify the x,y axis limits
        
        #plt.axis([-3*P.r0, 3*P.r0, -3*P.r0, 3*P.r0, -3*P.r0, 3*P.r0])
        # Draw line for the ground
        #self.ax.axes.set_xlim3d(left=-2*P.r0, right=2*P.r0) 
        #self.ax.axes.set_ylim3d(bottom=-2*P.r0, top=2*P.r0) 
        #self.ax.axes.set_zlim3d(bottom=-2*P.r0, top=2*P.r0) 
        #plt.show()
        
        #plt.xlim([-3*P.r0, 3*P.r0])
        #plt.ylim([-3*P.r0, 3*P.r0])
        #plt.zlim([-3*P.r0, 3*P.r0])
        #plt.plot([-2*P.ell, 2*P.ell], [0, 0], 'b--')
        # label axes
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.zlabel('z')

        ###################################################

        self.fig = plt.figure(figsize=(7,7), dpi=90)
        self.ax = self.fig.add_subplot(1, 1, 1,projection='3d')
        self.ax.set_xlim(( -10000, 10000))            
        self.ax.set_ylim((-10000, 10000))
        self.ax.set_zlim((-10000, 10000))

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        R=P.r0/2
        x = R * np.outer(np.cos(u), np.sin(v))
        y = R * np.outer(np.sin(u), np.sin(v))
        z = R * np.outer(np.ones(np.size(u)), np.cos(v))

        self.ax.plot_surface(x, y, z,  cmap='viridis', edgecolor='none')
 
        phi = 0
        theta = 0
        psi = 0
        r=P.r0
        sat_x = r*np.cos(theta)*np.cos(phi)
        sat_y = r*np.sin(theta)*np.cos(phi)
        sat_z = r*np.sin(phi)
 
        pn = sat_x
        pe = sat_y
        pd = sat_z
 
        pos=np.array([pn, pe, pd, phi, theta, psi])
 
        w=500 # WIDTH OF SATELLLITE

        self.v1=np.array([w/2, -w/2, -w/2]).T
        self.v2=np.array([-w/2, -w/2, -w/2]).T
        self.v3=np.array([-w/2, w/2, -w/2]).T
        self.v4=np.array([w/2, w/2, -w/2]).T

        self.v5=np.array([w/2, -w/2, w/2]).T
        self.v6=np.array([-w/2, -w/2, w/2]).T
        self.v7=np.array([-w/2, w/2, w/2]).T
        self.v8=np.array([w/2, w/2, w/2]).T

        # INERTIAL FRAME COORDINATE SYSTEM

        North_i=np.array([1, 0, 0]).T
        East_i=np.array([0, 1, 0]).T
        Down_i=np.array([0, 0, 1]).T

        # SHIFTING THE VERTICES FROM BODY TO INERTIAL

        pos_ned=np.array([pn, pe, pd]).T
        R1=self.RotationMatrix(phi,theta,psi)
        v1r=np.matmul(R1,self.v1)+pos_ned
        v2r=np.matmul(R1,self.v2)+pos_ned
        v3r=np.matmul(R1,self.v3)+pos_ned
        v4r=np.matmul(R1,self.v4)+pos_ned
        v5r=np.matmul(R1,self.v5)+pos_ned
        v6r=np.matmul(R1,self.v6)+pos_ned
        v7r=np.matmul(R1,self.v7)+pos_ned
        v8r=np.matmul(R1,self.v8)+pos_ned
        r_d=P.r_d0
        phi_d=P.phi_d0
        psi_d=0
        theta_d=P.theta_d0

        R2=self.RotationMatrix(phi, theta, psi)
        sat_x_d = r_d*np.cos(theta_d)*np.cos(phi_d)
        sat_y_d = r_d*np.sin(theta_d)*np.cos(phi_d)
        sat_z_d = r_d*np.sin(phi_d)
 
        pn_d = sat_x_d
        pe_d = sat_y_d
        pd_d = sat_z_d
        pos_dep = np.array([pn_d, pe_d, pd_d]).T

        w1r=np.matmul(R2,self.v1)+pos_dep
        w2r=np.matmul(R2,self.v2)+pos_dep
        w3r=np.matmul(R2,self.v3)+pos_dep
        w4r=np.matmul(R2,self.v4)+pos_dep
        w5r=np.matmul(R2,self.v5)+pos_dep
        w6r=np.matmul(R2,self.v6)+pos_dep
        w7r=np.matmul(R2,self.v7)+pos_dep
        w8r=np.matmul(R2,self.v8)+pos_dep

        # HANDLES FOR THE CUBE FACES

        f1_x=[v1r[0], v2r[0], v3r[0], v4r[0], v1r[0]]
        f1_y=[v1r[1], v2r[1], v3r[1], v4r[1], v1r[1]]
        f1_z=[v1r[2], v2r[2], v3r[2], v4r[2], v1r[2]]
        f2_x=[v5r[0], v6r[0], v7r[0], v8r[0], v5r[0]]
        f2_y=[v5r[1], v6r[1], v7r[1], v8r[1], v5r[1]]
        f2_z=[v5r[2], v6r[2], v7r[2], v8r[2], v5r[2]]
        f3_x=[v3r[0], v4r[0], v8r[0], v7r[0], v3r[0]]
        f3_y=[v3r[1], v4r[1], v8r[1], v7r[1], v3r[1]]
        f3_z=[v3r[2], v4r[2], v8r[2], v7r[2], v3r[2]]
        f4_x=[v2r[0], v1r[0], v5r[0], v6r[0], v2r[0]]
        f4_y=[v2r[1], v1r[1], v5r[1], v6r[1], v2r[1]]
        f4_z=[v2r[2], v1r[2], v5r[2], v6r[2], v2r[2]]

        ############

        g1_x=[w1r[0], w2r[0], w3r[0], w4r[0], w1r[0]]
        g1_y=[w1r[1], w2r[1], w3r[1], w4r[1], w1r[1]]
        g1_z=[w1r[2], w2r[2], w3r[2], w4r[2], w1r[2]]
        g2_x=[w5r[0], w6r[0], w7r[0], w8r[0], w5r[0]]
        g2_y=[w5r[1], w6r[1], w7r[1], w8r[1], w5r[1]]
        g2_z=[w5r[2], w6r[2], w7r[2], w8r[2], w5r[2]]
        g3_x=[w3r[0], w4r[0], w8r[0], w7r[0], w3r[0]]
        g3_y=[w3r[1], w4r[1], w8r[1], w7r[1], w3r[1]]
        g3_z=[w3r[2], w4r[2], w8r[2], w7r[2], w3r[2]]
        g4_x=[w2r[0], w1r[0], w5r[0], w6r[0], w2r[0]]
        g4_y=[w2r[1], w1r[1], w5r[1], w6r[1], w2r[1]]
        g4_z=[w2r[2], w1r[2], w5r[2], w6r[2], w2r[2]]

        self.face1, = self.ax.plot(f1_x, f1_y, f1_z, 'g', lw=2)
        self.face2, = self.ax.plot(f2_x, f2_y, f2_z, 'y', lw=2)
        self.face3, = self.ax.plot(f3_x, f3_y, f3_z, 'g', lw=2)
        self.face4, = self.ax.plot(f4_x, f4_y, f4_z, 'y', lw=2)

        ########################

        self.face1d, = self.ax.plot(g1_x, g1_y, g1_z, 'b', lw=2)
        self.face2d, = self.ax.plot(g2_x, g2_y, g2_z, 'b', lw=2)
        self.face3d, = self.ax.plot(g3_x, g3_y, g3_z, 'r', lw=2)
        self.face4d, = self.ax.plot(g4_x, g4_y, g4_z, 'r', lw=2)
 
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.set_title('Satellite-Earth System')


    def RotationMatrix(self,phi, theta, psi):
        R1=np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0,            0           , 1]])
        
        R2=np.array([[1,            0           , 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])
        R3=np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0,            0           , 1]])
        ROT=np.matmul(R3,np.matmul(R2,R1))
        return ROT

    def update(self, state):
        #r = state[0][0]  # Horizontal position of cart, m
        #theta = state[1][0]  # Angle of pendulum, rads
        #phi=  state[2][0]
        #x=r*np.cos(theta)*np.cos(phi)
        #y=r*np.cos(theta)*np.sin(phi)
        #z=r*np.sin(theta)
        # draw plot elements: cart, bob, rod

        self.draw_sat(state)
        #self.draw_bob(z, theta)
        #self.draw_rod(z, theta)
        #self.ax.axis('equal')
        #self.ay.axis('equal')
        #self.az.axis('equal')
        # Set initialization flag to False after first call
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.set_title('Satellite-Earth System')
        anim=animation.FuncAnimation(self.fig, self.draw_sat(state),frames=P.frames, interval=40, blit=False)
        #anim.save('basic_animation.mp4', fps=30)
        #f = r"C://Users/mukht/Desktop/animation.mp4" 
        #writervideo = animation.FFMpegWriter(fps=60) 
        #anim.save(f, writer=writervideo)
        #plt.show()
        #if self.flag_init == True:
        #    self.flag_init = False
    def satPoints(self,r,theta,phi):
        sat_x = r*np.cos(theta)*np.cos(phi)
        sat_y = r*np.sin(theta)*np.cos(phi)
        sat_z = r*np.sin(phi)
        return sat_x, sat_y, sat_z
    


    # Converting Quaternion to Euler 313
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
    


    def draw_sat(self, states):
        # specify bottom left corner of rectangle
        
        r = states[0]
        theta = states[2]
        phi = states[5]
 
        sat_x, sat_y, sat_z = self.satPoints(r, theta, phi)
        pn = sat_x
        pe = sat_y
        pd = sat_z
        y=self.quat2Eul313(states)
        #psi = y[0][0]
        #theta = y[1][0]
        #phi = y[2][0]
 
        pos_ned = np.array([pn, pe, pd]).T


        r_d = states[13]
        theta_d = states[15]
        phi_d = states[17]
 
        sat_x_d, sat_y_d, sat_z_d = self.satPoints(r_d, theta_d, phi_d)
        pn_d = sat_x_d
        pe_d = sat_y_d
        pd_d = sat_z_d
        R2=self.RotationMatrix(0, 0, 0)
        pos_dep = np.array([pn_d, pe_d, pd_d]).T

        w=500 # WIDTH OF SATELLLITE

        v1=np.array([w/2, -w/2, -w/2]).T
        v2=np.array([-w/2, -w/2, -w/2]).T
        v3=np.array([-w/2, w/2, -w/2]).T
        v4=np.array([w/2, w/2, -w/2]).T

        v5=np.array([w/2, -w/2, w/2]).T
        v6=np.array([-w/2, -w/2, w/2]).T
        v7=np.array([-w/2, w/2, w/2]).T
        v8=np.array([w/2, w/2, w/2]).T
 
        R1=self.RotationMatrix(0,0,0)
        
        v1r=np.matmul(R1,v1)+pos_ned
        v2r=np.matmul(R1,v2)+pos_ned
        v3r=np.matmul(R1,v3)+pos_ned
        v4r=np.matmul(R1,v4)+pos_ned
        v5r=np.matmul(R1,v5)+pos_ned
        v6r=np.matmul(R1,v6)+pos_ned
        v7r=np.matmul(R1,v7)+pos_ned
        v8r=np.matmul(R1,v8)+pos_ned

        ############

        w1r=np.matmul(R2,v1)+pos_dep
        w2r=np.matmul(R2,v2)+pos_dep
        w3r=np.matmul(R2,v3)+pos_dep
        w4r=np.matmul(R2,v4)+pos_dep
        w5r=np.matmul(R2,v5)+pos_dep
        w6r=np.matmul(R2,v6)+pos_dep
        w7r=np.matmul(R2,v7)+pos_dep
        w8r=np.matmul(R2,v8)+pos_dep
        #print(v1r)
# HANDLE FOR chief satellite , 4 FACES OF THE CUBE
        f1_x=[v1r[0][0], v2r[0][0], v3r[0][0], v4r[0][0], v1r[0][0]]
        f1_y=[v1r[0][1], v2r[0][1], v3r[0][1], v4r[0][1], v1r[0][1]]
        f1_z=[v1r[0][2], v2r[0][2], v3r[0][2], v4r[0][2], v1r[0][2]]
        f2_x=[v5r[0][0], v6r[0][0], v7r[0][0], v8r[0][0], v5r[0][0]]
        f2_y=[v5r[0][1], v6r[0][1], v7r[0][1], v8r[0][1], v5r[0][1]]
        f2_z=[v5r[0][2], v6r[0][2], v7r[0][2], v8r[0][2], v5r[0][2]]
        f3_x=[v3r[0][0], v4r[0][0], v8r[0][0], v7r[0][0], v3r[0][0]]
        f3_y=[v3r[0][1], v4r[0][1], v8r[0][1], v7r[0][1], v3r[0][1]]
        f3_z=[v3r[0][2], v4r[0][2], v8r[0][2], v7r[0][2], v3r[0][2]]
        f4_x=[v2r[0][0], v1r[0][0], v5r[0][0], v6r[0][0], v2r[0][0]]
        f4_y=[v2r[0][1], v1r[0][1], v5r[0][1], v6r[0][1], v2r[0][1]]
        f4_z=[v2r[0][2], v1r[0][2], v5r[0][2], v6r[0][2], v2r[0][2]]

        ######################

        self.face1.set_data(f1_x, f1_y)
        self.face1.set_3d_properties(f1_z)
        self.face2.set_data(f2_x, f2_y)
        self.face2.set_3d_properties(f2_z)
        self.face3.set_data(f3_x, f3_y)
        self.face3.set_3d_properties(f3_z)
        self.face4.set_data(f4_x, f4_y)
        self.face4.set_3d_properties(f3_z)

# HANDLE FOR deputy satellite , 4 FACES OF THE CUBE
        g1_x=[w1r[0][0], w2r[0][0], w3r[0][0], w4r[0][0], w1r[0][0]]
        g1_y=[w1r[0][1], w2r[0][1], w3r[0][1], w4r[0][1], w1r[0][1]]
        g1_z=[w1r[0][2], w2r[0][2], w3r[0][2], w4r[0][2], w1r[0][2]]
        g2_x=[w5r[0][0], w6r[0][0], w7r[0][0], w8r[0][0], w5r[0][0]]
        g2_y=[w5r[0][1], w6r[0][1], w7r[0][1], w8r[0][1], w5r[0][1]]
        g2_z=[w5r[0][2], w6r[0][2], w7r[0][2], w8r[0][2], w5r[0][2]]
        g3_x=[w3r[0][0], w4r[0][0], w8r[0][0], w7r[0][0], w3r[0][0]]
        g3_y=[w3r[0][1], w4r[0][1], w8r[0][1], w7r[0][1], w3r[0][1]]
        g3_z=[w3r[0][2], w4r[0][2], w8r[0][2], w7r[0][2], w3r[0][2]]
        g4_x=[w2r[0][0], w1r[0][0], w5r[0][0], w6r[0][0], w2r[0][0]]
        g4_y=[w2r[0][1], w1r[0][1], w5r[0][1], w6r[0][1], w2r[0][1]]
        g4_z=[w2r[0][2], w1r[0][2], w5r[0][2], w6r[0][2], w2r[0][2]]

 
       
        ###############################

        self.face1d.set_data(g1_x, g1_y)
        self.face1d.set_3d_properties(g1_z)
        self.face2d.set_data(g2_x, g2_y)
        self.face2d.set_3d_properties(g2_z)
        self.face3d.set_data(g3_x, g3_y)
        self.face3d.set_3d_properties(g3_z)
        self.face4d.set_data(g4_x, g4_y)
        self.face4d.set_3d_properties(g3_z)
 
 
        return self.face1, self.face2, self.face3, self.face4, self.face1d, self.face2d, self.face3d, self.face4d

    



        #corner = [x, y, z]
        #colors = np.empty([5,5,5] + [4], dtype=np.float32)
        #colors[:] = [1, 0, 0, 0.9]
        # create rectangle on first call, update on subsequent calls
       # if self.flag_init is True:
            # Create the Rectangle patch and append its handle
            # to the handle list
        #    self.handle.append(
         #       self.ax.plot_surface(X*2, Y*2, Z*2))
            
            # Add the patch to the axes
            #self.add_patch(self.handle[0])
        #else:
        #    self.handle[0].set_aa(corner)  # Update patch
       
    def draw_bob(self, z, theta):
        # specify center of circle
        x = z+(P.ell+P.radius)*np.sin(theta)
        y = P.gap+P.h+(P.ell+P.radius)*np.cos(theta)
        center = (x, y)
        # create circle on first call, update on subsequent calls
        if self.flag_init is True:
            # Create the CirclePolygon patch and append its handle
            # to the handle list
            self.handle.append(
                mpatches.CirclePolygon(center, radius=P.radius,
                                       resolution=15, fc='limegreen', ec='black'))
            # Add the patch to the axes
            self.ax.add_patch(self.handle[1])
        else:
            self.handle[1].xy = center

    def draw_rod(self, z, theta):
        # specify x-y points of the rod
        X = [z, z+P.ell*np.sin(theta)]
        Y = [P.gap+P.h, P.gap+P.h+P.ell*np.cos(theta)]
        # create rod on first call, update on subsequent calls
        if self.flag_init is True:
            # Create the line object and append its handle
            # to the handle list.
            line, = self.ax.plot(X, Y, lw=1, c='black')
            self.handle.append(line)
        else:
            self.handle[2].set_xdata(X)
            self.handle[2].set_ydata(Y)