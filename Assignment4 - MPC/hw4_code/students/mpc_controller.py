from linear_car_model import LinearCarModel
import numpy as np
import cvxpy as cp

class MPC:
    
    def __init__(self,  model : LinearCarModel, 
                        N     : int      ,
                        Q     : np.ndarray ,
                        R     : np.ndarray,
                        Qt    : np.ndarray,
                        solver: str = 'CLARABEL',
                        warm_start: bool = False):
        
        """
        Model Predictive Controller for a given linear system.
        
        :param model: The linear car model.
        :type model: LinearCarModel
        :param N: Prediction horizon.
        :type N: int
        :param Q: State cost matrix.
        :type Q: np.ndarray
        :param R: Input cost matrix.
        :type R: np.ndarray
        :param Qt: Terminal state cost matrix.
        :type Qt: np.ndarray
        :param solver: The solver to use for the optimization problem. Default is 'OSQP'.
        :type solver: str
        :param warm_start: Whether to use warm starting for the solver. Default is True.
        :type warm_start: bool
        :raises AssertionError: If the dimensions of Q, R, or Qt do not match the system dimensions.
        
        """
        
        
        #########################################################
        ## Setting up the MPC controller
        #########################################################
        
        self.model = model    # model of your system.
        self.N     = N        # Prediction horizon.
        self.n     = model.n  # State dimension.
        self.m     = model.m  # Input dimension.
        self.Q     = Q        # State cost matrix.
        self.R     = R        # Input cost matrix.
        self.Qt    = Qt       # Terminal state cost matrix.

        self.solver_type = solver # Solver type for the optimization problem.
        self.warm_start = warm_start # Warm start option for the solver.
        available_solvers = ['OSQP', 'MOSEK','CLARABEL']
        if solver not in available_solvers:
            raise ValueError(f"Unknown solver type: {solver}, Valid options are: {available_solvers}")

        # creating the bound vectors for  lb <= u <= ub.

        self.ubu = np.array([np.inf]*self.m)  # Upper bound on inputs.
        self.lbu = np.array([-np.inf]*self.m) # Lower bound on inputs.

        # creating the bound vectors for  lb <= x <= ub.
        self.ubx = np.array([np.inf]*self.n)  # Upper bound on states.
        self.lbx = np.array([-np.inf]*self.n) # Lower bound on states.

        self.is_setup = False

        # Check dimensions of Q, R, Qt
        assert Q.shape == (self.n, self.n), f"Q must be of shape ({self.n}, {self.n})"
        assert R.shape == (self.m, self.m), f"R must be of shape ({self.m}, {self.m})"
        assert Qt.shape == (self.n, self.n), f"Qt must be of shape ({self.n}, {self.n})"


    def add_upper_bound_on_state(self, state_index: int, upper_bound: float):
        """
        Add an upper bound constraint on a specific state variable.

        :param state_index: Index of the state variable to constrain.
        :type state_index: int
        :param upper_bound: Upper bound value.
        :type upper_bound: float
        """
        if state_index < 0 or state_index >= self.n:
            raise ValueError(f"state_index must be between 0 and {self.n - 1}")
        self.ubx[state_index] = upper_bound

    def add_lower_bound_on_state(self, state_index: int, lower_bound: float):
        """
        Add a lower bound constraint on a specific state variable.

        :param state_index: Index of the state variable to constrain.
        :type state_index: int
        :param lower_bound: Lower bound value.
        :type lower_bound: float
        """
        if state_index < 0 or state_index >= self.n:
            raise ValueError(f"state_index must be between 0 and {self.n - 1}")
        self.lbx[state_index] = lower_bound

    def add_upper_bound_on_input(self, input_index: int, upper_bound: float):
        """
        Add an upper bound constraint on a specific input variable.

        :param input_index: Index of the input variable to constrain.
        :type input_index: int
        :param upper_bound: Upper bound value.
        :type upper_bound: float
        """
        if input_index < 0 or input_index >= self.m:
            raise ValueError(f"input_index must be between 0 and {self.m - 1}")
        self.ubu[input_index] = upper_bound

    def add_lower_bound_on_input(self, input_index: int, lower_bound: float):
        """
        Add a lower bound constraint on a specific input variable.

        :param input_index: Index of the input variable to constrain.
        :type input_index: int
        :param lower_bound: Lower bound value.
        :type lower_bound: float
        """
        if input_index < 0 or input_index >= self.m:
            raise ValueError(f"input_index must be between 0 and {self.m - 1}")
        self.lbu[input_index] = lower_bound
    

    def setup_mpc_problem(self):
        """
        Set up the MPC optimization problem given the current state and reference trajectory.

        :param x0: Current state of the system.
        :type x0: np.ndarray
        :param xr: Reference trajectory over the prediction horizon.
        :type xr: np.ndarray
        
        """

        ###########################################################
        ## Extract system information
        ##########################################################
        A = self.model.A
        B = self.model.B
        n = self.n
        m = self.m
        N = self.N
        
        # Initialize cost matrices
        constraints = []
        cost = 0

        ##########################################################
        ## Define Optimization Problem Variables and Parameters.
        ##########################################################
        
        ## Variables definition
        x =  # todo: State variables over the horizon nxN
        u =  # todo: Control inputs over the horizon  mx(N-1)
        
        ## Define initial state parameter
        x_init = #todo: initial state parameter
        x_ff   = # todo:feedforward reference state parameter nxN
        u_ff   = # todo:feedforward reference input parameter mx(N-1)

        ## Define state error and input error
        e_x = # todo: create error as state difference
        e_u = # todo:create error as input difference


        ##########################################################
        ## Define Cost Function.
        ##########################################################
        for k in range(N-1):
            cost += # todo stage cost

        cost += # todo terminal cost

        ##########################################################
        ## Define Constraints.
        ##########################################################

        for k in range(N-1):

            # Define system dynamics constraints (e_x_{k+1} = A*e_x_k + B*e_u_k)
            constraints += [] # #todo:dynamic constraints

            # Define input constraints (self.lbu <= u_k <= self.ubu)
            for jj in range(self.m) :
                if self.lbu[jj] != -np.inf :
                    constraints += [] # todo: fill bounds
                if self.ubu[jj] != np.inf :
                    constraints += [] # todo: fill bounds

            # Define state constraints (self.lbx <= x_k <= self.ubx)
            for ii in range(self.n) :
                if self.lbx[ii] != -np.inf :
                    constraints += []# todo: fill bounds
                if self.ubx[ii] != np.inf :
                    constraints += []# todo: fill bounds

        # terminal constraint
        for ii in range(self.n) :
            if self.lbx[ii] != -np.inf :
                constraints += []# todo: fill bounds
            if self.ubx[ii] != np.inf :
                constraints += [] # todo: fill bounds

        # Define initial state constraint
        constraints += [] # todo: fill constraint

        ##########################################################
        ## Define Optimization Problem and store variables in self.
        ##########################################################


        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        self.x      = x
        self.u      = u
        self.e_u    = e_u
        self.e_x    = e_x
        self.x_ff   = x_ff
        self.u_ff   = u_ff
        self.x_init = x_init

        self.is_setup = True

    def solve_mpc(self, x0: np.ndarray, x_ff: np.ndarray, u_ff: np.ndarray):
        """
        Solve the MPC optimization problem for the current state and reference trajectory.

        :param x0: Current state of the system.
        :type x0: np.ndarray
        :param xr: Reference trajectory over the prediction horizon (N,n).
        :type xr: np.ndarray
        :param ur: Reference input over the prediction horizon (N-1,m).
        :type ur: np.ndarray
        :return: Optimal control input for the current time step.
        :rtype: np.ndarray
        """

        if not self.is_setup:
            raise ValueError("MPC problem is not set up. Call setup_mpc_problem() before solving.")

        # Set parameter values
        self.x_init.value = # todo:fill here
        self.x_ff.value   = # todo:fill here
        self.u_ff.value   = # todo:fill here

        # Solve the optimization problem
        self.problem.solve(solver=self.solver_type, warm_start=self.warm_start)

        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"MPC problem did not solve to optimality. Status is {self.problem.status}")

        # Return the first control input
        return # todo:return optimal input
    
    def get_solver_time(self):
        """
        Get the time taken by the solver to solve the last MPC problem.

        :return: Solver time in seconds.
        :rtype: float
        """
        if not self.is_setup:
            raise ValueError("MPC problem is not set up. Call setup_mpc_problem() before getting solver time.")
        
        return self.problem.solver_stats.solve_time 
    
    def get_solver_cost(self):
        """
        Get the cost of the last solved MPC problem.

        :return: Cost value.
        :rtype: float
        """
        if not self.is_setup:
            raise ValueError("MPC problem is not set up. Call setup_mpc_problem() before getting solver cost.")
        
        return self.problem.value




class MPCdual:
    
    def __init__(self,  model : LinearCarModel, 
                        N     : int      ,
                        Nd    :int       ,
                        Q     : np.ndarray ,
                        R     : np.ndarray,
                        Qt    : np.ndarray,
                        solver: str = 'CLARABEL',
                        warm_start: bool = False):
        
        """
        Model Predictive Controller for a given linear system.
        
        :param model: The linear car model.
        :type model: LinearCarModel
        :param N: Prediction horizon.
        :type N: int
        :param: Nd dual mode horizon
        :type Nd: int
        :param Q: State cost matrix.
        :type Q: np.ndarray
        :param R: Input cost matrix.
        :type R: np.ndarray
        :param Qt: Terminal state cost matrix.
        :type Qt: np.ndarray
        :param solver: The solver to use for the optimization problem. Default is 'OSQP'.
        :type solver: str
        :param warm_start: Whether to use warm starting for the solver. Default is True.
        :type warm_start: bool
        :raises AssertionError: If the dimensions of Q, R, or Qt do not match the system dimensions.
        
        """
        
        
        #########################################################
        ## Setting up the MPC controller
        #########################################################
        
        self.model = model    # model of your system.
        self.N     = N        # Prediction horizon.
        self.Nd    = Nd       # dual mode horizon
        self.n     = model.n  # State dimension.
        self.m     = model.m  # Input dimension.
        self.Q     = Q        # State cost matrix.
        self.R     = R        # Input cost matrix.
        self.Qt    = Qt       # Terminal state cost matrix.

        self.solver_type = solver # Solver type for the optimization problem.
        self.warm_start = warm_start # Warm start option for the solver.
        available_solvers = ['OSQP', 'MOSEK','CLARABEL']
        if solver not in available_solvers:
            raise ValueError(f"Unknown solver type: {solver}, Valid options are: {available_solvers}")

        # creating the bound vectors for  lb <= u <= ub.

        self.ubu = np.array([np.inf]*self.m)  # Upper bound on inputs.
        self.lbu = np.array([-np.inf]*self.m) # Lower bound on inputs.

        # creating the bound vectors for  lb <= x <= ub.
        self.ubx = np.array([np.inf]*self.n)  # Upper bound on states.
        self.lbx = np.array([-np.inf]*self.n) # Lower bound on states.

        self.is_setup = False

        # Check dimensions of Q, R, Qt
        assert Q.shape == (self.n, self.n), f"Q must be of shape ({self.n}, {self.n})"
        assert R.shape == (self.m, self.m), f"R must be of shape ({self.m}, {self.m})"
        assert Qt.shape == (self.n, self.n), f"Qt must be of shape ({self.n}, {self.n})"




    