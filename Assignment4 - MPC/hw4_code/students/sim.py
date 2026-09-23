from   matplotlib import pyplot as plt
import numpy as np
import cvxpy as cp

from linear_car_model import LinearCarModel
from road import RaceTrack
from mpc_controller import MPC
from scipy.linalg import sqrtm


def simulate_system_run(x0             : np.ndarray,  
                        system         : LinearCarModel, 
                        mpc_controller : MPC, 
                        lqr_controller : np.ndarray,
                        racetrack      : RaceTrack, 
                        raceline       : np.ndarray , 
                        ds             : float,
                        x_ff           : np.ndarray,
                        u_ff           : np.ndarray):
    """
    Run the simulation for a given initial state and number of steps.

    :param x0: Initial state vector
    :type x0: np.ndarray
    :param system: The linear car model system
    :type system: LinearCarModel
    :param controller: The MPC controller
    :type controller: MPCController
    :param racetrack: The racetrack object
    :type racetrack: RaceTrack
    :param raceline: The reference raceline to follow
    :type raceline: np.ndarray
    :param ds: Discretization step of the system
    :type ds: float
    :param x_ff: The feedforward state trajectory
    :type x_ff: np.ndarray
    :param u_ff: The feedforward control trajectory
    :type u_ff: np.ndarray
    """
    
    ######################################################################################
    # Plotting road and saving initialization data
    ######################################################################################

    ax_road                 = racetrack.plot_track()

    # compute raceline curvature 
    num_points              = len(raceline)
    heading                 = np.unwrap(np.arctan2(np.gradient(raceline[:,1]), np.gradient(raceline[:,0])))
    curvature               = np.gradient(heading) / ds
    raceline_curvature      = curvature
    raceline_length         = np.sum(np.linalg.norm(np.diff(raceline, axis=0), axis=1))
    s_ref                   = np.cumsum(np.hstack((0, np.linalg.norm(np.diff(raceline, axis=0), axis=1))))
    horizon                  = mpc_controller.N
    L                        = lqr_controller

    ######################################################################################
    # Compute Feedforward action
    ######################################################################################

    # extend reference beyond the raceline for the horizon of the controller
    x_ff_extd = np.vstack((x_ff, x_ff[:(horizon+1),:]))
    u_ff_extd = np.vstack((u_ff, u_ff[:horizon,:]))

    # pre allocate vectors to save simulation data
    x       = np.zeros((num_points, system.n))   # absolute state of the system
    u       = np.zeros((num_points-1, system.m)) # MPC control input to the system
    lqr_u   = np.zeros((num_points-1, system.m)) # LQR control input to the system
    e_x     = np.zeros((num_points, system.n))   # error state of the system
    e_u     = np.zeros((num_points-1, system.m)) # error input to the system
    s       = np.zeros(num_points)               # S coordinate along the raceline
    
    computational_time = np.zeros(num_points-1) # time taken to solve the MPC problem at each step
    mpc_cost           = np.zeros(num_points-1) # cost of the MPC problem at each step
    # Initialize Initial State and raceline position
    x[0]    = x0
    s[0]    = 0.

    ######################################################################################
    # Simulate Controller
    ######################################################################################
    bumps = []
    for i in range(num_points-1):

        u_ff_i = u_ff_extd[i:i+(horizon-1),:].T
        x_ff_i = x_ff_extd[i:i+(horizon),:].T

        # Compute control input
        u[i]      = mpc_controller.solve_mpc(x[i], x_ff_i, u_ff_i)  # input is u = e_u + u_ref
        lqr_u[i]  = -L @ (x[i] - x_ff[i]) + u_ff[i]    # input is u = LQR*(x - x_ref) + u_ref

        
        # update state and input errors
        e_x[i]    = x[i] - x_ff[i]
        e_u[i]    = u[i] - u_ff[i]
        
        # Update system state
        ki       = raceline_curvature[i]  # curvature at current position
        x[i + 1] = system.A @ x[i] + system.B @ u[i] + system.Bw @ np.array([ki])

        if i in [126, 590, 1026, 1475]:
            bumps.append(s[i])
            print(f"Bump at meter {s[i]:.2f} !")
            print(f"step {i}, s = {s[i]:.2f} m")
            x[i+1,0] += np.random.uniform(0, 0.9)
            x[i+1,1] += np.random.uniform(0, 0.2)

        # next s coordinate position
        s[i+1]    = s[i] + ds
        computational_time[i] = mpc_controller.get_solver_time()
        mpc_cost[i]           = mpc_controller.get_solver_cost()

    
    ######################################################################################
    # Plot Results
    ######################################################################################
    ED    = 0 # displacement error
    EPSI  = 1 # heading error
    VY    = 2 # lateral velocity
    R     = 3 # yaw rate
    DELTA = 4 # steering angle

    
    ######################################################################################
    # Plot absolute system state
    ######################################################################################
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
    axs[DELTA].legend()
    fig.suptitle(r'Absolute System States $x_k$ (Simulation vs Feedforward)')
    
    
    ######################################################################################
    # Plot absolute system input
    ######################################################################################
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
    
    ######################################################################################
    # Plot difference LQR and MPC
    ######################################################################################
    fig, ax = plt.subplots(2,1)
    ax[0].plot(s[:-1], lqr_u[:,0] - u[:,0], label=r'$u_{\delta}^{LQR} - u_{\delta}^{MPC}$')
    ax[1].plot(s[:-1], lqr_u[:,1] - u[:,1], label=r'$u_{r}^{LQR} - u_{r}^{MPC}$')
    ax[0].set_ylabel(r' $\Delta u_{\delta}$ [rad/s]')
    ax[1].set_ylabel(r' $\Delta u_{r}$ [Nm]')
    ax[1].set_xlabel('S coordinate [m]')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(r'Difference between LQR and MPC Control Inputs')
    plt.tight_layout()

    ######################################################################################
    # Plot State Error
    ######################################################################################
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

    ######################################################################################
    # Plot Input Error
    ######################################################################################
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
        

    fig2.suptitle(r'Control Input MPC Controller $mpc(x_k-x^{ff}_k)$')
    plt.tight_layout()


    ######################################################################################
    # Plot Road
    ######################################################################################

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

    ######################################################################################
    # Plot Computational Time
    ######################################################################################
    fig, ax = plt.subplots()
    ax.plot(s[:-1], computational_time*1000)
    ax.set_xlabel('S coordinate [m]')
    ax.set_ylabel('Computational Time [ms]')
    ax.grid()
    fig.suptitle('Computational Time per MPC Step')
    # average 
    avg_time = np.mean(computational_time)*1000
    # standard deviation
    std_time = np.std(computational_time)*1000
    # max time
    m_time  = np.max(computational_time)*1000

    ax.axhline(y=avg_time, color='r', linestyle='--', label=f'Average Time: {avg_time:.2f} ms')
    ax.fill_between(s[:-1], avg_time - std_time, avg_time + std_time, color='r', alpha=0.2, label=f'Std Dev: {std_time:.2f} ms')
    ax.axhline(y=avg_time + std_time, color='r', linestyle='--')
    ax.axhline(y=avg_time - std_time, color='r', linestyle='--')
    ax.axhline(y=m_time, color='k', linestyle='--', label=f'Max Time: {m_time:.2f} ms')
    
    ax.legend()

    ######################################################################################
    # Plot MPC cost
    ######################################################################################
    fig, ax = plt.subplots()
    ax.plot(s[:-1], mpc_cost)
    ax.set_xlabel('S coordinate [m]')
    ax.set_ylabel('MPC Cost')
    ax.grid()
    ax.set_yscale('log')
    fig.suptitle('MPC Cost per Step')
    # add bumps
    for i,bump in enumerate(bumps):
        if i == 0:
            ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2, label='Bumps')
        else:
            ax.axvline(x=bump, color='orange', linestyle='--', linewidth=2)




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
    s_ref                   = np.cumsum(np.hstack((0, np.linalg.norm(np.diff(raceline, axis=0), axis=1))))


    A  = system.A
    B  = system.B
    Bw = system.Bw
    C  = system.C

    # Define optimization variables
    x_ff = cp.Variable((num_points, system.n))
    u_ff = cp.Variable((num_points-1, system.m))
    y_ff = (C@x_ff.T).T

    # define cost function

    # define dynamic constraints
    constraints = []
    s           = 0
    for i in range(num_points - 1):

        k_i           = raceline_curvature[i]  # curvature at current position
        constraints += [x_ff[i + 1] == A @ x_ff[i] + B @ u_ff[i] + Bw @ np.array([k_i])]
        s           += ds

    # initial state constraint
    constraints += [x_ff[0] == np.zeros(system.n)] # set initial condition to all zeros

    # Define cost function (minimize control effort and deviation from raceline)
    Q_ff = np.diag([1000., 1000.])
    R_ff = np.diag([0.01, 0.001])
    cost = 0

    QQ = np.kron(np.eye(num_points-1), Q_ff)
    RR = np.kron(np.eye(num_points-1), R_ff)

    cost += cp.quad_form(cp.vec(y_ff[:-1,:],order = "C"), QQ)*ds + cp.quad_form(cp.vec(u_ff,order = "C"), RR)*ds

    # Terminal cost
    cost += cp.quad_form(y_ff[-1], Q_ff)*ds

    # add constraint on steering angle 
    constraints += [cp.abs(x_ff[:,-1]) <= np.deg2rad(25)]  # steering angle limits

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
    ax2[0].set_ylabel(r' $u_{delta}$ [rad/s]')
    ax2[0].grid()

    # differential torque
    ax2[1].plot(s[:-1],u_ff[:,1])
    ax2[1].set_xlabel('S coordinate [m]')
    ax2[1].set_ylabel(r' $u_{r}$ [Nm]')
    ax2[1].grid()


    fig2.suptitle(r'Feedforward Control Inputs $u_{\delta}, u_{r}$')
    plt.tight_layout()



# def simulate_system_run_with_kalman_filter(x0 : np.ndarray,  system : LinearCarModel, controller, Sigma_q : np.ndarray, Sigma_r : np.ndarray, P : np.ndarray):
#     """
#     Run the simulation for a given initial state and number of steps.
    
#     :param x0: Initial state vector
#     :type x0: np.ndarray
#     :param n_steps: Number of simulation steps
#     :type n_steps: int
#     :returns: State and input trajectories over the simulation horizon.
#     :rtype: Tuple[np.ndarray, np.ndarray]
#     """
    
#     num_points    = 800
#     racetrack     = RaceTrack()
#     qp_solver     = QPSolver(racetrack, num_points=num_points)
#     raceline      = qp_solver.compute_minimum_acceleration_raceline()
#     ax_road       = racetrack.plot_track()
#     ds            = system.ds
#     L             = controller
    
#     # resample the reference trajectory to have it uniformly spaced in s
#     raceline_length = np.sum(np.linalg.norm(np.diff(raceline, axis=0), axis=1))
#     num_points      = int(np.ceil(raceline_length/ds))
#     s_ref           = np.arange(0, raceline_length, ds)

    
#     raceline_x    = np.interp(s_ref, np.cumsum(np.hstack((0, np.linalg.norm(np.diff(raceline, axis=0), axis=1)))), raceline[:,0])
#     raceline_y    = np.interp(s_ref, np.cumsum(np.hstack((0, np.linalg.norm(np.diff(raceline, axis=0), axis=1)))), raceline[:,1])
#     raceline      = np.column_stack((raceline_x, raceline_y)) # now raceline is sampled at equally spaced s values according to the ds of the system

#     # compute raceline curvature 
#     heading                 = np.unwrap(np.arctan2(np.gradient(raceline[:,1]), np.gradient(raceline[:,0])))
#     curvature               = np.gradient(heading) / ds
#     raceline_curvature      = curvature

#     x_ff, u_ff = compute_feedforward_action(system, raceline=raceline, ds=ds)

#     # Initialize state and input vector to save variables of simulation
#     e_x_pred     = np.zeros((num_points, system.n)) # absolute predicted state of the system
#     e_x          = np.zeros((num_points, system.n)) # absolute true state of the system
#     x_pred       = np.zeros((num_points, system.n)) # predicted state of the system
#     x            = np.zeros((num_points, system.n))
#     u            = np.zeros((num_points, system.m)) # LQR control input to the system
#     s            = np.zeros(num_points)             # S coordinate along the raceline


#     # Initialize real and predicted state of the system
#     e_x_pred[0]  = x0 - x_ff[0]  # initial predicted state
#     e_x[0]       = x0 - x_ff[0]  + np.random.multivariate_normal(np.zeros(system.n), P) # initial true state with some noise
#     x_pred[0]    = e_x_pred[0] + x_ff[0]
#     x[0]         = e_x[0] + x_ff[0]

#     si           = 0.

#     for i in range(num_points-1):

#         # Update state prediction based on the current measurment of the system (Kalman update)
#         v_k      = np.random.multivariate_normal(np.zeros(system.p), Sigma_r)    # measurment noise
#         y_i      = system.C @ e_x[i] + v_k                                       # take measurement with noise from real system
        
#         ## Kalman filter Update
        
#         # Innovation covariance
#         S = system.C @ P @ system.C.T + Sigma_r
#         # Kalman gain
#         K = P @ system.C.T @ np.linalg.inv(S)
#         # Update state estimate
#         e_y       = y_i - system.C @ e_x_pred[i]
#         e_x_pred[i] = e_x_pred[i] + K @ e_y
#         # Update covariance
#         I = np.eye(e_x_pred[i].shape[0])
#         P = (I - K @ system.C) @ P
        

#         ## Compute control input
#         # Compute control input based on predicted state
#         u[i]      = - L @ (e_x_pred[i]) + u_ff[i]
        
#         # Propagate the predicted dynamics
#         ki        =  raceline_curvature[i]
#         x_pred[i] = e_x_pred[i] + x_ff[i]
#         # predicted_model 
#         x_pred[i + 1] = system.A @ x_pred[i] + system.B @ u[i] + system.Bw @ np.array([ki])
#         P             = system.A @ P @ system.A.T + Sigma_q
#         e_x_pred[i+1] = x_pred[i+1] - x_ff[i+1]

#         # Propagate real system with noise
#         w_k      = np.random.multivariate_normal(np.zeros(system.n), Sigma_q)
#         x[i]     = e_x[i] + x_ff[i]
#         x[i + 1] = system.A @ x[i] + system.B @ u[i] + system.Bw @ np.array([ki]) + w_k
#         e_x[i+1] = x[i+1] - x_ff[i+1]
        
#         # next s coordinate position
#         s[i+1]    = s[i] + ds

    
#     # plot system states
#     fig, ax = plt.subplots(5,1)
#     ax[0].set_ylabel(r' $e_{d}$ [m]')
#     ax[1].set_ylabel(r' $e_{\psi}$ [rad]')
#     ax[2].set_ylabel(r' $\dot{e}_{\psi}$ [rad/s]')
#     ax[3].set_ylabel(r' $v_{x}$ [m/s]')
#     ax[4].set_ylabel(r' $\delta$ [rad]')
#     ax[4].set_xlabel('S coordinate [m]')
#     ax[0].grid()
#     ax[1].grid()
#     ax[2].grid()
#     ax[3].grid()
#     ax[4].grid()    

#     ax[0].plot(s,x_pred[:,0], color='blue', label='Predicted state')
#     ax[1].plot(s,x_pred[:,1], color='blue', label='Predicted state')
#     ax[2].plot(s,x_pred[:,2], color='blue', label='Predicted state')
#     ax[3].plot(s,x_pred[:,3], color='blue', label='Predicted state')
#     ax[4].plot(s,x_pred[:,4], color='blue', label='Predicted state')

#     ax[0].plot(s,x[:,0], color='blue', label='True state')
#     ax[1].plot(s,x[:,1], color='blue', label='True state')
#     ax[2].plot(s,x[:,2], color='blue', label='True state')
#     ax[3].plot(s,x[:,3], color='blue', label='True state')
#     ax[4].plot(s,x[:,4], color='blue', label='True state')
    

#     # plot feed forward
#     ax[0].plot(s,x_ff[:,0], color='red', label='Feedforward')
#     ax[1].plot(s,x_ff[:,1], color='red', label='Feedforward')
#     ax[2].plot(s,x_ff[:,2], color='red', label='Feedforward')
#     ax[3].plot(s,x_ff[:,3], color='red', label='Feedforward')
#     ax[4].plot(s,x_ff[:,4], color='red', label='Feedforward')
#     ax[0].legend()
#     fig.suptitle('System States (Simulation vs Feedforward)')
#     plt.tight_layout()

#     # plot state error
#     fig, ax = plt.subplots(5,1)
#     ax[0].set_ylabel(r' $e_{d}$ [m]')
#     ax[1].set_ylabel(r' $e_{\psi}$ [rad]')
#     ax[2].set_ylabel(r' $\dot{e}_{\psi}$ [rad/s]')
#     ax[3].set_ylabel(r' $v_{y}$ [m/s]')
#     ax[4].set_ylabel(r' $\delta$ [rad]')
#     ax[4].set_xlabel('S coordinate [m]')
#     ax[0].grid()
#     ax[1].grid()
#     ax[2].grid()
#     ax[3].grid()
#     ax[4].grid()
#     ax[0].plot(s,e_x[:,0], color='green', label='True Error')
#     ax[1].plot(s,e_x[:,1], color='green', label='True Error')
#     ax[2].plot(s,e_x[:,2], color='green', label='True Error')
#     ax[3].plot(s,e_x[:,3], color='green', label='True Error')
#     ax[4].plot(s,e_x[:,4], color='green', label='True Error')

#     ax[0].plot(s,e_x_pred[:,0], color='green', label='Predicted Error')
#     ax[1].plot(s,e_x_pred[:,1], color='green', label='Predicted Error')
#     ax[2].plot(s,e_x_pred[:,2], color='green', label='Predicted Error')
#     ax[3].plot(s,e_x_pred[:,3], color='green', label='Predicted Error')
#     ax[4].plot(s,e_x_pred[:,4], color='green', label='Predicted Error')

#     ax[0].legend()
#     fig.suptitle('State Error (Simulation - Feedforward)')
#     plt.tight_layout()

#     # # steering angle
#     # fig2, ax2 = plt.subplots(2,1)
#     # ax2[0].plot(s,e_u[:,0])
#     # ax2[0].set_xlabel('S coordinate [m]')
#     # ax2[0].set_ylabel(r' $\dot{\delta}_{cmd}$ [rad/s]')
#     # ax2[0].grid()

#     # # differential torque
#     # ax2[1].plot(s,e_u[:,1])
#     # ax2[1].set_xlabel('S coordinate [m]')
#     # ax2[1].set_ylabel(r' $T_{cmd}$ [Nm]')
#     # ax2[1].grid()
    

#     # fig2.suptitle('Control Input LQR Controller (Simulation - Feedforward)')
#     plt.tight_layout()

#     # absolute input 
#     fig, ax = plt.subplots(2,1)
#     ax[0].plot(s,u[:,0], color='blue', label='Simulation')
#     ax[0].plot(s,u_ff[:,0], color='red', label='Feedforward')
#     ax[0].set_xlabel('S coordinate [m]')
#     ax[0].set_ylabel(r' $\dot{\delta}_{cmd}$ [rad/s]')
#     ax[0].grid()
#     ax[0].legend()
#     ax[1].plot(s,u[:,1], color='blue', label='Simulation')
#     ax[1].plot(s,u_ff[:,1], color='red', label='Feedforward')
#     ax[1].set_xlabel('S coordinate [m]')
#     ax[1].set_ylabel(r' $T_{cmd}$ [Nm]')
#     ax[1].grid()
#     ax[1].legend()
#     fig.suptitle('Control Input (Simulation vs Feedforward)')


#     plt.tight_layout()


#     # plot vehcile path

#     X = np.zeros(num_points)
#     Y = np.zeros(num_points)
    
#     for ii in range(len(s)) :
        
#         si = s[ii]

#         xi        = raceline[ii,0]
#         yi        = raceline[ii,1]
#         heading_i = heading[ii]

#         X[ii] = xi - x[ii,0]*np.sin(heading_i)
#         Y[ii] = yi + x[ii,0]*np.cos(heading_i)

#     # plot raceline
#     ax_road.plot(raceline[:,0], raceline[:,1], 'k', label='Racing Line', linewidth=3)
#     # add s coordinate close to the racline every 10 meters
#     for si in range(0, int(raceline_length), 10):
#         xi = np.interp(si, s_ref, raceline[:,0])
#         yi = np.interp(si, s_ref, raceline[:,1])
#         ax_road.text(xi, yi, f's={si:.0f}m', color='red', fontsize=8)

#     # plot vehicle path
#     ax_road.plot(X, Y, 'b-', label='Vehicle Path')

#     ax_road.set_xlabel('X [m]')
#     ax_road.set_ylabel('Y [m]')
#     ax_road.legend()
#     ax_road.set_title('Vehicle Trajectory on Road')


        
    
