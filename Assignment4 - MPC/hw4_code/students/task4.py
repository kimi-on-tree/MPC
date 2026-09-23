from linear_car_model    import LinearCarModel 
from sim                 import simulate_system_run, compute_feedforward_action, plot_feedforward
from road                import RaceTrack
from mpc_controller      import MPC

import numpy as np
import matplotlib.pyplot as plt
import os


#-----------------------------------------------------------------
# Do not touch this block
#------------------------------------------------------------------
# np.random.seed(200)
np.random.seed(200)
dir               = os.path.dirname(os.path.abspath(__file__))
use_saved_values = True

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
raceline      = np.load(os.path.join(dir, "raceline.npy"))

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

x_ff = np.load(os.path.join(dir, "x_ff.npy"))
u_ff = np.load(os.path.join(dir, "u_ff.npy"))    
s    = np.cumsum(np.hstack((0,np.ones(num_points-1)*ds)))

#-----------------------------------------------------------------
# Start of the assignment
#------------------------------------------------------------------

####################################
# Q3
####################################

Q = np.diag([10, 1, 0.1, 0.1, 1])
R = np.diag([0.1, 0.1])


L = system.get_lqr_controller(Q, R)
P = system.get_lqr_cost_matrix(Q, R)


##  Define MPC controller
mpc_controller = MPC(model      = system, 
                     N          = 20, 
                     Q          = Q, 
                     R          = R, 
                     Qt         = P*1, 
                     warm_start = True, 
                     solver     = 'CLARABEL')
mpc_controller.setup_mpc_problem()

x0 = np.array([0.9, -0.1, 0, 0, 0])

simulate_system_run(x0         = x0, 
                    system     = system, 
                    mpc_controller = mpc_controller, 
                    lqr_controller = L,
                    racetrack  = racetrack, 
                    raceline   = raceline, 
                    ds         = ds,
                    x_ff       = x_ff,
                    u_ff       = u_ff)


# ####################################
# # Q5 - Q6 
# ####################################

# Q = np.diag([100, 10, 10, 10, 1])
# R = np.diag([10, 0.1])
# lambda_val = 100.


# L = system.get_lqr_controller(Q, R)
# P = system.get_lqr_cost_matrix(Q, R)


# ##  Define MPC controller
# mpc_controller = MPC(model      = system, 
#                      N          = 20, 
#                      Q          = Q, 
#                      R          = R, 
#                      Qt         = P*lambda_val, 
#                      warm_start = True, 
#                      solver     = 'MOSEK')

# TODO:add constraints
mpc_controller.add_lower_bound_on_input() 

# TODO:add constraints
mpc_controller.add_lower_bound_on_state() 


# leave this line here!
mpc_controller.setup_mpc_problem()


# x0 = np.array([0.9, -0.1, 0, 0, 0])

# simulate_system_run(x0         = x0, 
#                     system     = system, 
#                     mpc_controller = mpc_controller, 
#                     lqr_controller = L,
#                     racetrack  = racetrack, 
#                     raceline   = raceline, 
#                     ds         = ds,
#                     x_ff       = x_ff,
#                     u_ff       = u_ff)



####################################
# Q7
####################################

# Q = np.diag([10, 1, 0.1, 0.1, 1])
# R = np.diag([0.1, 0.1])


# L = system.get_lqr_controller(Q, R)
# P = system.get_lqr_cost_matrix(Q, R)

# N1 = 20
# N2 = 40
# N3 = 80

# solver1 = 'CLARABEL'
# solver2 = 'MOSEK' 

# ##  Define MPC controller
# mpc_controller = MPC(model      = system, 
#                      N          = 20, 
#                      Q          = Q, 
#                      R          = R, 
#                      Qt         = P*1, 
#                      warm_start = True, 
#                      solver     = solver1)



# mpc_controller.setup_mpc_problem()

# x0 = np.array([0.9, -0.1, 0, 0, 0])

# simulate_system_run(x0         = x0, 
#                     system     = system, 
#                     mpc_controller = mpc_controller, 
#                     lqr_controller = L,
#                     racetrack  = racetrack, 
#                     raceline   = raceline, 
#                     ds         = ds,
#                     x_ff       = x_ff,
#                     u_ff       = u_ff)





plt.show()