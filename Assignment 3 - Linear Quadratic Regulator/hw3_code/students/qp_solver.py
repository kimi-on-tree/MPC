from   road import RaceTrack
import numpy as np
import cvxpy as cp

class QPSolver:
    def __init__(self, racetrack : RaceTrack, num_points: int = 1000):
        
        self.racetrack = racetrack
        self.num_points = num_points
        self.positions = None
        self.s_values = np.linspace(0, 1, self.num_points)
        self.s_delta = np.diff(self.s_values)[0]

    def compute_minimum_acceleration_raceline(self):
        
        """
        Computes a minimum-curvature racing line within track boundaries using
        convex optimization. Currently returns the vehicle positions and the center line
        offset for the racing line.
        """
        vehicle_positions = cp.Variable((self.num_points, 2))
        n                 = cp.Variable(self.num_points)
        
        # Track width constraints
        track_constraints = [cp.abs(n) <= self.racetrack.width/2]
        
        # Relate positions to centerline offset
        for i, s in enumerate(self.s_values):
            track_constraints += [
                vehicle_positions[i, :] == self.racetrack.position(s) + n[i] * self.racetrack.normal(s)
            ]

        ## add obstacle avoidance constraints here if needed
        for i in range(int(self.num_points*0.4), int(self.num_points*0.6)):
            track_constraints += [
                n[i] <= -1.5
            ]

        D2 = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points):
            D2[i, i] = -2
            D2[i, (i - 1) % self.num_points] = 1
            D2[i, (i + 1) % self.num_points] = 1

        # acceleration = cp.sum(cp.norm(D2 @ vehicle_positions, axis=1))
        acceleration = cp.sum_squares(D2 @ vehicle_positions)

        min_acc_problem = cp.Problem(cp.Minimize(acceleration), track_constraints)

        min_acc_problem.solve(solver=cp.MOSEK)
        self.positions = vehicle_positions.value

        return self.positions

