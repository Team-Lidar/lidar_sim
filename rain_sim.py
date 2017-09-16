"""
Adapted from "Animation of elastic collisions with Gravity"
author: Jake Vanderplas
license: BSD

Edit by Khan Hossein Shamsuttoha
This was an attempt to simulate particle collision between two sets of particles. For each particle, you can control the masa, size, x, y, Vx and Vy
The particles collide with each other and an object changing velocity. All collision assumed are elastic collisions.
The time parameter dt is important. The collision calculations are done after time period dt. If dt is too large we might simulate particles passing through each other without interacting.
The boundaries of the world are cyclical. Meaning the world loops around.
For now it is a 2D simulation. But for later revisions we could easily add z and Vz
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import pdb;

class RainParticleBox:
    """ 
    init_state is an array with N number of arrays each representing a particle. Each particle is described by its position and speed vectors:      
       [[x1, y1, vx1, vy1, mass, size, is_rain_particle],
        [x2, y2, vx2, vy2, mass, size, is_rain_particle],
        ...               
        N times
       ]
    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1, 0.05, 0.04,1],
                               [-0.5, 0.5, 0.5, 0.5, 0.05, 0.04,1],
                               [-0.5, -0.5, -0.5, 0.5, 0.05, 0.04],0],
                 bounds = [-10, 10, -2, 2],
                 size = 0.04,
                 M = 0.05,
                 G = 9.8):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        self.update_colliding_particles()
        self.check_boundary_crossings()
        self.check_target_hits()
        self.add_gravity()
        self.add_wind()
        
    def update_colliding_particles(self):
		#Update positions multiplying time and velocity and adding it to current displacement
        self.state[:, :2] += dt * self.state[:, 2:4]
        #Find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        #Update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.state[i1,4]
            m2 = self.state[i2,4]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:4]
            v2 = self.state[i2, 2:4]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:4] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:4] = v_cm - v_rel * m1 / (m1 + m2) 
            
              
    def check_boundary_crossings(self):
		#Check for crossing boundary. for now considering a looped environment
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)
        
		#If crossed, loop the world
        self.state[crossed_x1, 0] *= -1
        self.state[crossed_x2, 0] *= -1
        self.state[crossed_y1, 1] *= -1
        self.state[crossed_y2, 1] *= -1
		        		    
    def check_target_hits(self):
		#Check for hitting target. for now considering a looped environment
        about_to_hit_x_1 = self.state[:, 0] - self.state[:, 5]
        about_to_hit_x_2 = self.state[:, 0] + self.state[:, 5]
        x_match_1 = (about_to_hit_x_1 > -0.05)
        x_match_2 = (about_to_hit_x_1 < 0.05)
        x_match = np.logical_and(x_match_1,x_match_2)
        
        y_match_1 = (self.state[:, 1] > -1)
        y_match_2 = (self.state[:, 1] < 1)
        y_match = np.logical_and(y_match_1,y_match_2) 
               
        hitting_target = np.logical_and(x_match,y_match)
		#If hitting complete elastic collision
        self.state[hitting_target, 2] *= -1
         
    def add_gravity(self):
		#Add acceleration due to gravity
        not_terminal_yet = np.where(self.state[:, 3] > -1)
        for i in zip(not_terminal_yet):
			self.state[i,3] -= self.state[i,4] * self.G * dt #add acceleration due to gravity
        
    def add_wind(self):
		# Add wind
        not_terminal_yet = np.where(self.state[:, 2] < 3)
        for i in zip(not_terminal_yet):
			self.state[i,2] += 0.25 #Some arbitraty acceleration in the x direction	
#------------------------------------------------------------

def init():
    """initialize animation"""
    global box, rect, rain_data, light_data
    rain_particles.set_data([], [])
    light_particles.set_data([], [])
    rect.set_edgecolor('none')
    return rain_particles, light_particles, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig, rain_data, light_data
    box.step(dt)
	
	#Need to clean the hardcoded values out
    ms_1 = int(fig.dpi * 2 * box.state[0,5] * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
             
    ms_2 = int(fig.dpi * 2 * box.state[149,5] * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    #Update pieces of the animation
    rect.set_edgecolor('k')
    
    #Would have liked to use the is_rain property. Could not figure out the iteration over ndarray
    np.copyto(rain_data,box.state[:100,:])
    np.copyto(light_data,box.state[100:,:])

    rain_particles.set_data(rain_data[:, 0], rain_data[:, 1])
    rain_particles.set_markersize(ms_1)
    light_particles.set_data(light_data[:,0], light_data[:,1])
    light_particles.set_markersize(ms_2)
    return rain_particles, light_particles, rect

def setup_animation():
	global rain_particles, light_particles, rect, fig, box, dt, ax, rain_data, light_data
	#Set up figure and animation
	fig = plt.figure()
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
	                     xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))
	
	#rain_particles holds the locations of the rain particles
	rain_particles, = ax.plot([], [], 'bo', ms=6)
	#light_particles 2 holds the loction of another set of particles
	light_particles, = ax.plot([], [], 'go', ms=6)
	
	rain_data = np.empty(shape=(100, 7))
	light_data = np.empty(shape=(50, 7))
	
	# rect is the box edge, identifies the edge of the environment
	rect = plt.Rectangle(box.bounds[::2],
	                     box.bounds[1] - box.bounds[0],
	                     box.bounds[3] - box.bounds[2],
	                     ec='none', lw=2, fc='none')
	ax.add_patch(rect)
	# line is a 1d object for reflection
	line = plt.vlines(0, -1, 1, 'r')
	ani = animation.FuncAnimation(fig, animate, frames=200, interval=5, blit=True, init_func=init)
	plt.show()

#Set up Boundary
bounds = [-10, 10, -2, 2]

#Set up initial state of rain particles
np.random.seed(0)
init_state = -0.5 + np.random.sample((150, 7)) # initialize all position and velocity vectors to values between -0.5 to 0.5 
init_state[:100, 0:1] *= bounds[1] - bounds[0] # Normalise to distribute particles all accross the x
init_state[:100, 1:2] *= bounds[3] - bounds[2] # Normalise to distribute particles all accross the y
init_state[:100, 3:4] -= 0.5 # Want particles to have negative Y velocity. Rain particles falling down 
init_state[:100, 3:4] *= 2	# Give it some initial downward velocity
init_state[:100, 4] = 0.05 # Assume all particles have same mass, i.e rain particles
init_state[:100, 5] = 0.04 # Assume all particles have same size, i.e rain particles
init_state[:100, 6] = 1 # Assume are rain particles


#Set up initial state of the light particles
init_state[100:, 0:1] *= bounds[1] - bounds[0] # Normalise to distribute particles all accross the x
init_state[100:, 1:2] *= bounds[3] - bounds[2] # Normalise to distribute particles all accross the y
init_state[100:, 2] = 3 # Want particles to have negative Y velocity. Rain particles falling down 
init_state[100:, 4] = 0.005 # Assume all particles have same mass, i.e rain particles
init_state[100:, 5] = 0.02 # Assume all particles have same size, i.e rain particles
init_state[100:, 6] = 0 # Assume are rain particles


#Create environment using the initial state of the particles
box = RainParticleBox(init_state, size=0.02)
dt = 1. / 300 # 300fps

setup_animation()

#------------------------------------------------------------

		
