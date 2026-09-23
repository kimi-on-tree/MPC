import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from   racetrack_tools import  RaceTrack
import matplotlib.pyplot as plt
from   racing_line import compute_racing_line, compute_lap_time


#####################################################################
# race track initialization
####################################################################

num_points  = 1000                            # number of discretized points to find the best raceline
name        = "montecarlo"                    # name of the racetrack
racetrack   = RaceTrack.get_racetrack(name)   # racetrack object (contains track parameters such as track_width)
mu          = 1.2                             # friction coefficient
g           = 9.81                            # gravity
s_delta     = racetrack.length / (num_points - 1)


s_values = np.linspace(0, racetrack.length, num_points)              # define a grid of points
center   = np.array([racetrack.position(s) for s in s_values])       
normal   = np.array([racetrack.normal_vector(s) for s in s_values])


# positions = center
# TODO: Implement compute_racing_line
# positions = compute_racing_line(racetrack, num_points, cost="norm")  # Eq. (4)
positions = compute_racing_line(racetrack, num_points, cost="sq")  # Eq. (7) for Q4


# TODO: Implement lap time calculation
lap_time_race = compute_lap_time(positions, g, mu, s_delta)
lap_time_center = compute_lap_time(center, g, mu, s_delta)
# minutes, seconds = divmod(lap_time, 60)
# print("Lap time:", f"{int(minutes):02d}:{seconds:06.3f}")
for label, t in [("Centerline", lap_time_center), ("Racing line", lap_time_race)]:
    minutes, seconds = divmod(t, 60)
    print(f"{label} lap time: {int(minutes):02d}:{seconds:06.3f}")

plt.close("all")
plt.figure(figsize=(12, 10))
plt.plot(center[:,0] + racetrack.track_width/2 * normal[:,0], center[:,1] + racetrack.track_width/2 * normal[:,1], "k")
plt.plot(center[:,0] - racetrack.track_width/2 * normal[:,0], center[:,1] - racetrack.track_width/2 * normal[:,1], "k")
plt.plot(center[:, 0], center[:, 1], "r--", alpha=0.5, label="Centerline")
plt.plot(positions[:, 0], positions[:, 1], "b", alpha=0.5, label="Racing line")
plt.title("Circuit de Monaco")
plt.legend()
plt.show()
