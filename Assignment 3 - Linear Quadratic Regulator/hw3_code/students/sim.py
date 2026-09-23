from matplotlib import pyplot as plt
import numpy as np 
from linear_car_model import LinearCarModel
from matplotlib.patches import Polygon
from road import RaceTrack
from qp_solver import QPSolver
import cvxpy as cp




def simulate_system_run(x0 : np.ndarray,  system : LinearCarModel, controller, racetrack : RaceTrack, raceline : np.ndarray , ds : float):
    """
    Run the simulation for a given initial state and number of steps.

    :param x0: Initial state vector
    :type x0: np.ndarray
    :param system: The linear car model system
    :type system: LinearCarModel
    :param controller: The state feedback controller gain matrix (L) for the feedback controller u = -Lx
    :type controller: np.ndarray
    :param racetrack: The racetrack object
    :type racetrack: RaceTrack
    :param raceline: The reference raceline to follow
    :type raceline: np.ndarray
    :param ds: Discretization step of the system
    :type ds: float
    """
    
    ######################################################################################
    # Plotting road and saving initialization data
    ######################################################################################

    ax_road       = racetrack.plot_track()

    # compute raceline curvature 
    num_points              = len(raceline)
    heading                 = np.unwrap(np.arctan2(np.gradient(raceline[:,1]), np.gradient(raceline[:,0])))
    curvature               = np.gradient(heading) / ds
    raceline_curvature      = curvature
    raceline_length         = np.sum(np.linalg.norm(np.diff(raceline, axis=0), axis=1))
    s_ref                   = np.cumsum(np.hstack((0, np.linalg.norm(np.diff(raceline, axis=0), axis=1))))
    L                       = controller

    ######################################################################################
    # Compute Feedforward action
    ######################################################################################

    x_ff, u_ff = compute_feedforward_action(system, raceline=raceline, ds=ds)

    # pre allocate vectors to save simulation data
    x       = np.zeros((num_points, system.n))   # absolute state of the system
    u       = np.zeros((num_points-1, system.m)) # LQR control input to the system
    e_x     = np.zeros((num_points, system.n))   # error state of the system
    e_u     = np.zeros((num_points-1, system.m)) # error input to the system
    s       = np.zeros(num_points)               # S coordinate along the raceline
    
    # Initialize Initial State and raceline position
    x[0]    = x0
    s[0]    = 0.

    ######################################################################################
    # Simulate Controller
    ######################################################################################
    bumps = []
    for i in range(num_points-1):
        
        # Compute control input
        u[i]      = - L @ (x[i]-x_ff[i]) + u_ff[i]
        
        # update state and input errors
        e_x[i]    = x[i] - x_ff[i]
        e_u[i]    = u[i] - u_ff[i]
        
        # Update system state
        ki        =  raceline_curvature[i]
        x[i + 1] = system.A @ x[i] + system.B @ u[i] + system.Bw @ np.array([ki])

        if np.random.rand() < 0.005:
            bumps.append(s[i])
            print(f"Adding bump at step {i}")
            x[i+1,0] += np.random.uniform(-0.3, 0.3)
            x[i+1,1] += np.random.uniform(-0.05, 0.05)

        # next s coordinate position
        s[i+1]    = s[i] + ds

    
    ######################################################################################
    # Plot Results
    ######################################################################################
    ED    = 0 # displacement error
    EPSI  = 1 # heading error
    VY    = 2 # lateral velocity
    R     = 3 # yaw rate
    DELTA = 4 # steering angle
    
    # plot system states
    fig, axs = plt.subplots(5,1)
    axs[ED].set_ylabel(r' $e_{d}$ [m]')
    axs[EPSI].set_ylabel(r' $e_{\psi}$ [rad]')
    axs[VY].set_ylabel(r' $v_{y}$ [rad/s]')
    axs[R].set_ylabel(r' $r$ [m/s]')
    axs[DELTA].set_ylabel(r' $\delta$ [rad]')
    axs[DELTA].set_xlabel('S coordinate [m]')
    axs[ED].grid()
    axs[EPSI].grid()
    axs[VY].grid()
    axs[R].grid()
    axs[DELTA].grid()

    axs[ED].plot(s,x[:,ED], color='blue', label='Simulation')
    axs[EPSI].plot(s,x[:,EPSI], color='blue', label='Simulation')
    axs[VY].plot(s,x[:,VY], color='blue', label='Simulation')
    axs[R].plot(s,x[:,R], color='blue', label='Simulation')
    axs[DELTA].plot(s,x[:,DELTA], color='blue', label='Simulation')

    # add bumps 
    for ax in axs:
        for i,bump in enumerate(bumps):
            if i == 0:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2, label='Bumps')
            else:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2)
    

    # plot feed forward
    axs[ED].plot(s,x_ff[:,ED], color='red', label=r'Feedforward $x^{ff}$')
    axs[EPSI].plot(s,x_ff[:,EPSI], color='red', label=r'Feedforward $x^{ff}$')
    axs[VY].plot(s,x_ff[:,VY], color='red', label=r'Feedforward $x^{ff}$')
    axs[R].plot(s,x_ff[:,R], color='red', label=r'Feedforward $x^{ff}$')
    axs[DELTA].plot(s,x_ff[:,DELTA], color='red', label=r'Feedforward $x^{ff}$')
    axs[DELTA].axhline(y= np.deg2rad(35), color='k', linestyle='--', label='Linear Steering limits')
    axs[DELTA].axhline(y=-np.deg2rad(35), color='k', linestyle='--')
    axs[ED].legend()
    fig.suptitle(r'Absolute System States $x_k$ (Simulation vs Feedforward)')
    plt.tight_layout()

    # plot state error
    fig, axs = plt.subplots(5,1)
    axs[ED].set_ylabel(r' $e_{d}$ [m]')
    axs[EPSI].set_ylabel(r' $e_{\psi}$ [rad]')
    axs[VY].set_ylabel(r' $v_{y}$ [rad/s]')
    axs[R].set_ylabel(r' $r$ [m/s]')
    axs[DELTA].set_ylabel(r' $\delta$ [rad]')
    axs[DELTA].set_xlabel('S coordinate [m]')
    axs[ED].grid()
    axs[EPSI].grid()
    axs[VY].grid()
    axs[R].grid()
    axs[DELTA].grid()
    axs[ED].plot(s,e_x[:,ED], color='green', label=r'Error (x_k - x_k^{ff})')
    axs[EPSI].plot(s,e_x[:,EPSI], color='green', label=r'Error (x_k - x_k^{ff})')
    axs[VY].plot(s,e_x[:,VY], color='green', label=r'Error (x_k - x_k^{ff})')
    axs[R].plot(s,e_x[:,R], color='green', label=r'Error (x_k - x_k^{ff})')
    axs[DELTA].plot(s,e_x[:,DELTA], color='green', label=r'Error (x_k - x_k^{ff})')


    fig.suptitle(r'State Error $(x_k-x^{ff}_k)$')

   # add bumps
    for ax in axs:
        for i,bump in enumerate(bumps):
            if i == 0:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2, label='Bumps')
            else:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()

    plt.tight_layout()

    # steering angle
    fig2, ax2s = plt.subplots(2,1)
    ax2s[0].plot(s[:-1],e_u[:,0])
    ax2s[0].set_xlabel('S coordinate [m]')
    ax2s[0].set_ylabel(r' $u_{\delta}$ [rad/s]')
    ax2s[0].grid()

    # differential torque
    ax2s[1].plot(s[:-1],e_u[:,1])
    ax2s[1].set_xlabel('S coordinate [m]')
    ax2s[1].set_ylabel(r' $u_{r}$ [Nm]')
    ax2s[1].grid()

    for ax in ax2s:
        for i,bump in enumerate(bumps):
            if i == 0:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2, label='Bumps')
            else:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2)
    if len(bumps) > 0:
        ax2s[0].legend()
        ax2s[1].legend()
        

    fig2.suptitle(r'Control Input LQR Controller $-L(x_k-x^{ff}_k)$')
    plt.tight_layout()

    # absolute input
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(s[:-1], u[:, 0], color='blue', label='Simulation')
    axs[0].plot(s[:-1], u_ff[:, 0], color='red', label='Feedforward')
    axs[0].set_xlabel('S coordinate [m]')
    axs[0].set_ylabel(r' $u_{\delta}$ [rad/s]')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(s[:-1], u[:, 1], color='blue', label=r'Simulation $-L(x_k - x_k^{ff}) + u_k^{ff}$')
    axs[1].plot(s[:-1], u_ff[:, 1], color='red', label=r'Feedforward $u_k^{ff}$')
    axs[1].set_xlabel('S coordinate [m]')
    axs[1].set_ylabel(r' $u_{r}$ [Nm]')
    axs[1].grid()
    axs[1].legend()
    fig.suptitle(r'Control Input $-L(x_k - x_k^{ff}) + u_k^{ff}$ (feedback+feedforward)')
    plt.tight_layout()

    # absolute input with bumps
    for ax in axs:
        for i,bump in enumerate(bumps):
            if i == 0:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2, label='Bumps')
            else:
                ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2)
    if len(bumps) > 0:
        axs[0].legend()
        axs[1].legend()

    # plot vehicle path
    X = np.zeros(num_points)
    Y = np.zeros(num_points)
    
    for ii in range(len(s)) :
        
        si = s[ii]

        xi        = raceline[ii,0]
        yi        = raceline[ii,1]
        heading_i = heading[ii]

        X[ii] = xi - x[ii,0]*np.sin(heading_i)
        Y[ii] = yi + x[ii,0]*np.cos(heading_i)

    # plot raceline
    ax_road.plot(raceline[:,0], raceline[:,1], 'k', label='Racing Line', linewidth=3)
    # add s coordinate close to the racline every 10 meters
    for ii in range(0, int(len(s)), 100):
        si = s[ii]
        xi = np.interp(si, s_ref, raceline[:,0])
        yi = np.interp(si, s_ref, raceline[:,1])
        ax_road.text(xi, yi, f's={si:.0f}m', color='red', fontsize=8)

    # plot vehicle path
    ax_road.plot(X, Y, 'b-', label='Vehicle Path')

    ax_road.set_xlabel('X [m]')
    ax_road.set_ylabel('Y [m]')
    ax_road.legend()
    ax_road.set_title('Vehicle Trajectory on Road')




def compute_feedforward_action(system : LinearCarModel, raceline : np.ndarray, ds : float ):
    """
    Compute the feedforward action for a given raceline and system.
    :param system: The linear car model system
    :type system: LinearCarModel
    :param raceline: The reference raceline to follow
    :type raceline: np.ndarray
    :param ds: Discretization step of the system
    :type ds: float
    :returns: The feedforward state and input trajectories.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    # compute raceline curvature 
    num_points              = raceline.shape[0]
    heading                 = np.unwrap(np.arctan2(np.gradient(raceline[:,1]), np.gradient(raceline[:,0])))
    curvature               = np.gradient(heading) / ds
    raceline_curvature      = curvature
    
    # # REMOVE UPON COMPLETION
    # raise NotImplementedError("TODO: implement feedforward action computation")
    

    A  = system.A
    B  = system.B
    Bw = system.Bw
    C  = system.C

    # Define optimization variables
    x_ff = cp.Variable((num_points, system.n))
    #TODO: create variable num_points x system.n
    u_ff = cp.Variable((num_points - 1, system.m))
    # TODO : create variable num_points-1 x system.m
    y_ff = x_ff @ C.T
    # TODO : create output as a function of system state (hint: use the matrix C)

    # define cost function

    # define dynamic constraints
    constraints = []
    for i in range(num_points - 1):

        k_i          = raceline_curvature[i]
        constraints += [
            x_ff[i+1] == x_ff[i] @ A.T + u_ff[i] @ B.T + k_i * Bw.flatten()
            ]#TODO: add dynamics constraints

    # initial state constraint
    constraints += [x_ff[0] == np.zeros(system.n)]#TODO:Add initial state constraints

    # Define cost function (minimize control effort and deviation from raceline)
    Q_ff = np.diag([1000, 1000]) # TODO: define cost matrix
    R_ff = np.diag([0.1, 0.001]) # TODO: define cost matrix
    cost = 0
    for i in range(num_points-1):
        y_i = x_ff[i] @ C.T
        cost += cp.quad_form(y_i, Q_ff) + cp.quad_form(u_ff[i], R_ff)
        # TODO:define cost on the state and input
    
        # Terminal cost
    y_N = x_ff[-1] @ C.T
    cost += cp.quad_form(y_N, Q_ff)

    # add constraint on steering angle 
    delta_limit = np.deg2rad(25)
    for i in range(num_points):
        constraints += [cp.abs(x_ff[i, 4]) <= delta_limit]
        
    # Add constraints and objective to the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver = cp.MOSEK) 

    return x_ff.value, u_ff.value



def plot_feedforward(x_ff, u_ff, s) :
    """
    
    PLot the feedforward action computed by the compute_feedforward_action function.
    :param x_ff: The feedforward state trajectory
    :type x_ff: np.ndarray
    :param u_ff: The feedforward input trajectory
    :type u_ff: np.ndarray
    :param s: The s coordinate along the raceline
    :type s: np.ndarray
    """
    
    fig, ax = plt.subplots(5,1)
    ax[0].set_ylabel(r' $e_{d}$ [m]')
    ax[1].set_ylabel(r' $e_{\psi}$ [rad]')
    ax[2].set_ylabel(r' $v_{y}$ [rad/s]')
    ax[3].set_ylabel(r' $r$ [m/s]')
    ax[4].set_ylabel(r' $\delta$ [rad]')
    ax[4].set_xlabel('S coordinate [m]')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    ax[4].grid()    

    # plot feed forward
    ax[0].plot(s,x_ff[:,0], color='red', label='Feedforward')
    ax[1].plot(s,x_ff[:,1], color='red', label='Feedforward')
    ax[2].plot(s,x_ff[:,2], color='red', label='Feedforward')
    ax[3].plot(s,x_ff[:,3], color='red', label='Feedforward')
    ax[4].plot(s,x_ff[:,4], color='red', label='Feedforward')

    ax[4].axhline(y= np.deg2rad(25), color='k', linestyle='--', label='Steering limits')
    ax[4].axhline(y=-np.deg2rad(25), color='k', linestyle='--') 
    ax[0].legend()
    fig.suptitle('Feedforward State Trajectories')
    plt.tight_layout()


    # steering angle
    fig2, ax2 = plt.subplots(2,1)
    ax2[0].plot(s[:-1],u_ff[:,0])
    ax2[0].set_xlabel('S coordinate [m]')
    ax2[0].set_ylabel(r' $u_{\delta}$ [rad/s]')
    ax2[0].grid()

    # differential torque
    ax2[1].plot(s[:-1],u_ff[:,1])
    ax2[1].set_xlabel('S coordinate [m]')
    ax2[1].set_ylabel(r' $u_{r}$ [Nm]')
    ax2[1].grid()


    fig2.suptitle(r'Feedforward Control Inputs $u_{\delta}, u_{r}$')
    plt.tight_layout()