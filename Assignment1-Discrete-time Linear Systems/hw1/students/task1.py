import numpy as np

from linear_car_model    import LinearCarModel
from controller import Controller
from sim_env    import EmbeddedSimEnvironment

# Create pendulum and controller objects
vref    = 20

#TODO: Inside the class LinearCarModel, implement the method to save the system dynamics _compute_system_matrices. 
vehicle = LinearCarModel(dt=0.1, velocity_ref=vref)
ctl     = Controller()

# Get the system discrete-time dynamics
vehicle.c2d()
Ad, Bd, Cd, Dd = vehicle.get_discrete_dynamics()

# Plot poles and zeros
vehicle.poles_zeros(Ad, Bd, Cd, Dd)

# Get control gains
ctl.set_system(A=Ad, B=Bd, C=Cd, D=Dd)  # TODO: Set the discrete time system matrices
K = ctl.get_closed_loop_gain(p=[0.90, 0.91]) # TODO:change the poles to respect the constraints

# Set the desired reference based on the dock position and zero velocity on docked position
x_ref = np.array([[1.0, 0.0]]).T
x0    = np.array([[-1.0, 0.0]]).T

# Initialize simulation environment
ctl.set_reference(x_ref)
sim_env = EmbeddedSimEnvironment(model      = vehicle,
                                 dynamics   = vehicle.linearized_discrete_dynamics,
                                 controller = ctl,
                                 time       = 40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()

# Disturbance effect
vehicle.set_disturbance()
sim_env = EmbeddedSimEnvironment(model=vehicle,
                                 dynamics=vehicle.linearized_discrete_dynamics,
                                 controller=ctl,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()

# Activate feed-forward gain
# TODO: To activate the integral action you need to change class Controller() first !
ctl.activate_integral_action(dt=0.1, ki= 0.01) # TODO:Set a good value for the intergal gain Ki
t, y, u = sim_env.run(x0)
sim_env.visualize()
