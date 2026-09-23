import numpy as np
import control as ctrl
import matplotlib.pyplot as plt



class LinearCarModel:
    
    def __init__(self, Lf : float = 1.55, 
                       Lr:float = 1.35, 
                       mass : float= 1500, 
                       max_acc : float = 3.0, 
                       velocity_ref: float = 20.0,
                       dt: float = 0.1):

        """
        Linear vehicle model for a car.

        :param Lf: Distance from the center of gravity to the front axle (m)
        :type Lf: float
        :param Lr: Distance from the center of gravity to the rear axle (m)
        :type Lr: float
        :param mass: Mass of the vehicle (kg)
        :type mass: float
        :param max_acc: Maximum acceleration (m/s^2)
        :type max_acc: float
        :param velocity_ref: Reference velocity for the vehicle (m/s)
        :type velocity_ref: float
        """

        self.Lf      :float = Lf
        self.Lr      :float = Lr
        self.mass    :float = mass
        self.max_acc :float = max_acc
        self.L       :float = Lf + Lr
        self.dt      :float = dt

        self.w       :float = 0.0
        self.v_ref   :float = velocity_ref  # reference velocity    
        

        # continuous time dynamics
        self.A : np.ndarray = None
        self.B : np.ndarray = None
        self.C : np.ndarray = None
        self.D : np.ndarray = None
        

        # discrete time dynamics
        self.Ad : np.ndarray = None
        self.Bd : np.ndarray = None
        self.Cd : np.ndarray = None
        self.Dd : np.ndarray = None


        self.A, self.B, self.C, self.D = self._compute_system_matrices(v_ref=velocity_ref)

    

    def _compute_system_matrices(self,v_ref : float = 20.0):
        """
        Compute the system matrices A, B, C, D for the linearized continuous-time vehicle model.

        :param v_ref: Reference velocity for the vehicle (m/s)
        :type v_ref: float
        """
        
        k = self.Lr / self.L
        
        # # TODO: set the system matrices A, B, C, D and remove error (vehicle properties are available under self.Lf, self.Lr, self.mass, self.max_acc, self.v_ref)
        # raise NotImplementedError("System matrices computation not implemented yet")
        
        A = np.array([[0.0,1.0],
                      [0.0,0.0]])
        
        B = np.array([[0.0],
                      [v_ref*k]])
        
        C = np.array([[1.0,0.0]])

        D = np.array([[0.0]])

        return A, B, C, D
    
    

    def get_discrete_dynamics(self):
        """
        Get the discrete-time dynamics matrices Ad, Bd, Cd, Dd.

        :return: Discrete-time dynamics matrices
        :rtype: tuple
        """
        if self.Ad is None or self.Bd is None:
            raise ValueError("Discrete dynamics not set. Call c2d() first.")
        
        return self.Ad, self.Bd, self.Cd, self.Dd   

    def c2d(self):
        """
        Convert the continuous-time system to a discrete-time system.

        :param Ts: Sampling time (s)
        :type Ts: float
        :return: Discrete-time system matrices A_d, B_d, C_d, D_d
        :rtype: tuple
        """
        
        # create a continuous time system in state space form
        continuous_system = ctrl.ss(self.A, self.B, self.C, self.D)
        # create a discrete time system in state space form
        discrete_system   = ctrl.c2d(continuous_system, self.dt, method='zoh')
        # extract the discrete time matrices
        ( Ad_list , Bd_list , Cd_list , Dd_list ) = ctrl.ssdata ( discrete_system  )
        
        # convret the list to numpy arrays
        self.Ad = np.array ( Ad_list )
        self.Bd = np.array ( Bd_list )
        self.Cd = np.array ( Cd_list )
        self.Dd = np.array ( Dd_list )

    def set_discrete_dynamics(self, Ad, Bd):
        """
        Helper function to populate discrete-time dynamics

        :param Ad: discrete-time transition matrix
        :type Ad: np.ndarray
        :param Bd: discrete-time control input matrix
        :type Bd: np.ndarray
        """

        self.Ad = Ad
        self.Bd = Bd

    def set_disturbance(self):
        """
        Activate disturbance acting on the system
        """
        self.w = -0.02

    def disable_disturbance(self):
        """
        Disable the disturbance effect.
        """
        self.w = 0.0

    def get_disturbance(self):
        """
        Return the disturbance value

        :return: disturbance value
        :rtype: float
        """
        return self.w

    def linearized_discrete_dynamics(self, x:np.ndarray, u:np.ndarray):
        """
        Method to propagate discrete-time dynamics for Astrobee

        :param x: state
        :type x: np.ndarray
        :param u: control input
        :type u: np.ndarray
        :return: state after dt seconds
        :rtype: np.ndarray
        """

        if self.Ad is None or self.Bd is None:
            raise ValueError("Discrete dynamics not set. Call c2d() first.")
        
     
        x_next = self.Ad @ x + self.Bd @ u
        
        # constant disturbance
        if self.w != 0.0:
            Bw = np.zeros((2, 1))
            Bw[1, 0] = 1
            x_next = x_next - Bw * self.w

        return x_next

    def poles_zeros(self, Ad, Bd, Cd, Dd):
        """
        Plots the system poles and zeros.

        :param Ad: state transition matrix
        :type Ad: np.ndarray
        :param Bd: control matrix
        :type Bd: np.ndarray
        :param Cd: state-observation matrix
        :type Cd: np.ndarray
        :param Dd: control-observation matrix
        :type Dd: np.ndarray
        """
        # dt == 0 -> Continuous time system
        # dt != 0 -> Discrete time system
        sys = ctrl.ss(Ad, Bd, Cd, Dd, dt=self.dt)
        ctrl.pzmap(sys)
        plt.show()
        return
    


    

    
    
