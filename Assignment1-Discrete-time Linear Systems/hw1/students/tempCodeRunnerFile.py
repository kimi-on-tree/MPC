# Disturbance effect
# vehicle.set_disturbance()
# sim_env = EmbeddedSimEnvironment(model=vehicle,
#                                  dynamics=vehicle.linearized_discrete_dynamics,
#                                  controller=ctl,
#                                  time=40.0)
# t, y, u = sim_env.run(x0)
# sim_env.visualize()

# # Activate feed-forward gain
# # TODO: To activate the integral action you need to change class Controller() first !
# ctl.activate_integral_action(dt=0.1, ki= 0.000) # TODO:Set a good value for the intergal gain Ki
# t, y, u = sim_env.run(x0)
# sim_env.visualize()
