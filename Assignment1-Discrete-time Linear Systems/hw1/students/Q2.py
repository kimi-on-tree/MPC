import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from linear_car_model import LinearCarModel

vehicle = LinearCarModel(dt=0.1, velocity_ref=20)

# Q2a: compare
Ts = vehicle.dt
b = vehicle.B[1, 0]
Ad_m = np.array([[1.0, Ts], [0.0, 1.0]])
Bd_m = np.array([[Ts**2/2 * b], [Ts * b]])

vehicle.c2d()
Ad, Bd, Cd, Dd = vehicle.get_discrete_dynamics()
print("Q2a Ad match:", np.allclose(Ad_m, Ad))
print("Q2a Bd match:", np.allclose(Bd_m, Bd))

# Q2b: continuous
sys_c = ctrl.ss(vehicle.A, vehicle.B, vehicle.C, vehicle.D)
print("Continuous poles:", ctrl.poles(sys_c))
print("Continuous zeros:", ctrl.zeros(sys_c))
ctrl.pzmap(sys_c)
plt.show()

# Q2b: discrete
sys_d = ctrl.ss(Ad, Bd, Cd, Dd, dt=vehicle.dt)
print("Discrete poles:", ctrl.poles(sys_d))
print("Discrete zeros:", ctrl.zeros(sys_d))
vehicle.poles_zeros(Ad, Bd, Cd, Dd)