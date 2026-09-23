from linear_car_model import LinearCarModel 
import numpy as np
import matplotlib.pyplot as plt
from sim  import simulate_system_run, compute_feedforward_action, plot_feedforward
from qp_solver import QPSolver
from road import RaceTrack

np.random.seed(100)

###################################
# Linear system definition
###################################
v_ref  = 20.0 # m/s
ds     = 0.2  # m
system = LinearCarModel( ds = ds, velocity_ref=v_ref) # todo:fill the system dynamics
system.c2d()


##########################################################
# Create Racetrack and raceline (DO NOT MODIFY THIS BLOCK)
##########################################################
#----------------------------------------------------------
num_points    = 800
racetrack     = RaceTrack()
qp_solver     = QPSolver(racetrack, num_points=num_points)
raceline      = qp_solver.compute_minimum_acceleration_raceline()
ds            = system.ds

# resample the reference trajectory to have it uniformly spaced in s
raceline_length = np.sum(np.linalg.norm(np.diff(raceline, axis=0), axis=1))
num_points      = int(np.ceil(raceline_length/ds))
s_ref           = np.arange(0, raceline_length, ds)

raceline_x    = np.interp(s_ref, np.cumsum(np.hstack((0, np.linalg.norm(np.diff(raceline, axis=0), axis=1)))), raceline[:,0])
raceline_y    = np.interp(s_ref, np.cumsum(np.hstack((0, np.linalg.norm(np.diff(raceline, axis=0), axis=1)))), raceline[:,1])
raceline      = np.column_stack((raceline_x, raceline_y)) # now raceline is sampled at equally spaced s values according to the ds of the system
num_points    = raceline.shape[0]
#----------------------------------------------------------

#############################################
# Design Feedforward State-Input Trajectories
#############################################

x_ff, u_ff = compute_feedforward_action(system, raceline=raceline, ds=ds)
s          = np.cumsum(np.hstack((0,np.ones(num_points-1)*ds)))
# plotting feed-forward action
# plot system states
plot_feedforward(x_ff, u_ff, s)
plt.show()

# ####################################
# # Design LQR Controller
# ####################################

# #TODO: simulate the system for the different LQR setups (fillin the the different cost matrices Q and R)

# # Q1 = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
# # R1 = np.diag([100, 100])

# # Q2 = np.diag([100, 100, 0.1, 0.1, 1])
# # R2 = np.diag([1, 1])

# Q3 = np.diag([100, 100, 0.1, 0.1, 1])
# R3 = np.diag([0.1, 0.1])


# np.random.seed(100)
# L  = system.get_lqr_controller(Q3, R3)
# x0 = np.array([0.9, -0.1, 0, 0, 0])



# simulate_system_run(x0         = x0, 
#                     system     = system, 
#                     controller = L, 
#                     racetrack  = racetrack, 
#                     raceline   = raceline, 
#                     ds         = ds )


##########################################################################
# Design LQR Controller (Bryson rule) 
# (you can comment the previous simulate_system_run to avoid multiple plots)
##########################################################################
 

# Take maximum value of each state and control in the feedforward trajectory
# normalization coefficients
q1   = np.max(np.abs(x_ff[:, 0]))   # e_d
q2   = np.max(np.abs(x_ff[:, 1]))   # e_psi
q3   = np.max(np.abs(x_ff[:, 2]))   # v_y
q4   = np.max(np.abs(x_ff[:, 3]))   # r
q5   = np.max(np.abs(x_ff[:, 4]))   # delta

r1 = np.max(np.abs(u_ff[:, 0]))     # u_delta
r2 = np.max(np.abs(u_ff[:, 1]))     # u_r
 
# tuning coefficents
p1 = 100.0
p2 = 100.0
p3 = 1.0
p4 = 1.0
p5 = 50.0

l1 = 1.0
l2 = 1.0




Q = np.diag([p1/q1**2, p2/q2**2, p3/q3**2, p4/q4**2, p5/q5**2])
R = np.diag([l1/r1**2, l2/r2**2])

np.random.seed(100)
L  = system.get_lqr_controller(Q, R)
x0 = np.array([0.9, -0.1, 0, 0, 0])

simulate_system_run(x0         = x0, 
                    system     = system, 
                    controller = L, 
                    racetrack  = racetrack, 
                    raceline   = raceline, 
                    ds         = ds )


plt.show()