from   racetrack_tools import RaceTrack
import numpy as np
import cvxpy as cp


def compute_racing_line(racetrack: RaceTrack, num_points: int = 1000, cost: str = "norm"):
    """
    Computes a minimum-acceleration racing line within track boundaries using
    convex optimization. Returns the vehicle positions.

    cost:
        "norm" - Eq. (4): sum of ||a_i||_2 * ds
        "sq"   - Eq. (7): sum of ||a_i||_2^2 * ds
    """
    racetrack  = racetrack
    s_values   = np.linspace(0, racetrack.length, num_points)
    center     = np.array([racetrack.position(t) for t in s_values])
    normal = np.array([racetrack.normal_vector(s) for s in s_values])
    num_points = num_points
    s_delta    = racetrack.length / (num_points - 1)

    # TODO: Define cvxpy variables for vehicle positions (num_points x 2) and
    # the centerline offsets (vector of length num_points)
    positions =  cp.Variable((num_points, 2))
    n         = cp.Variable(num_points)

    # TODO: Add constraint on centerline offset
    track_constraints = [cp.abs(n) <= racetrack.track_width / 2]

    # TODO: Relate positions to centerline offset (Eq. (1))
    for i, s in enumerate(s_values):
        track_constraints += [positions[i, :] == center[i, :] + normal[i, :] * n[i]]

    # TODO: Define acceleration objective (Eq. (4) / Eq. (7))
    acc = (positions[2:, :] - 2 * positions[1:-1, :] + positions[:-2, :]) / s_delta**2
    if cost == "norm":
        # Eq. (4)
        acceleration = cp.sum(cp.norm(acc, axis=1)) * s_delta
    elif cost == "sq":
        # Eq. (7)
        acceleration = cp.sum_squares(acc) * s_delta
    else:
        raise ValueError('cost must be "norm" or "sq"')

    # TODO: Define cvxpy problem and solve it using MOSEK
    min_acc_problem = cp.Problem(cp.Minimize(acceleration), track_constraints)
    min_acc_problem.solve(solver=cp.MOSEK)
    
    print("Solver status:", min_acc_problem.status)
    return positions.value


def compute_lap_time(positions, g, mu, s_delta):
    v = np.diff(positions, n=1, axis=0)[:-1] / s_delta
    a = np.diff(positions, n=2, axis=0) / s_delta ** 2

    # TODO: Compute the curvature at each index i (Eq. (5))
    curvature = np.abs(a[:,1]*v[:,0] - a[:,0]*v[:,1])/(v[:, 0]**2 + v[:, 1]**2) ** 1.5
    
    # TODO: Compute the absolute velocity at each index i
    V = np.sqrt(mu*g/curvature)

    # TODO: Compute the lap time
    time_deltas = s_delta / V
    lap_time = np.sum(time_deltas)

    return lap_time
