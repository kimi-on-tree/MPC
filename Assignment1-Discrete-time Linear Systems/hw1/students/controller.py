import numpy as np
from control.matlab import place


class Controller:

    def __init__(self):
        """
        Linear Controller class. Implements the controller:
        u_t = - L @ x_t + lr @ r
        """

        # Hint:
        self.p1 = 0.98
        self.p2 = 0.97

        self.poles  = [self.p1, self.p2]
        self.L      = np.zeros((1, 2))
        self.i_term = 0.0
        self.Ki     = 0.0

        self.dt           = None
        self.use_integral = False

        print(self)                             # You can comment this line

    def __str__(self):
        return """
            Controller class. Implements the controller:
            u_t = - L @ x_t + lr @ r
        """

    def set_system(self, A=np.zeros((2, 2)), B=np.zeros((2, 1)),
                   C=np.ones((1, 2)), D=np.zeros((2, 1))):
        """
        Set system matrices.

        :param A: state space A matrix
        :type A: np.ndarray
        :param B: state space B matrix
        :type B: np.ndarray
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def get_closed_loop_gain(self, p=None):
        """
        Get the closed loop gain for the specified poles.

        :param p: pole list, defaults to self.p
        :type p: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if p is None:
            p = self.poles

        A = self.A.tolist()
        B = self.B.tolist()

        L = place(A, B, p)
        self.L[0, 0] = L[0, 0]
        self.L[0, 1] = L[0, 1]

        return self.L

    def set_poles(self, p, p2=None, p3=None, p4=None):
        """
        Set closed loop poles. If 'p' is a list of poles, then the remaining
        inputs are ignored. Otherwise, [p,p2,p3,p4] are set as poles.

        :param p: pole 1 or pole list
        :type p: list or scalar
        :param p2: pole 2, defaults to None
        :type p2: scalar, optional
        """

        if isinstance(p, list):
            self.poles = p
        else:
            self.p1 = p
            self.p2 = p2
            self.p3 = p3
            self.p4 = p4
            self.poles = [self.p1, self.p2, self.p3, self.p4]

    def get_feedforward_gain(self, L=None):
        """
        Get the feedforward gain lr.

        :param L: close loop gain, defaults to None
        :type L: list, optional
        """

        if L is None:
            L = self.L

        self.lr = 1.0 / (self.C @ np.linalg.inv(np.eye(2) - (self.A - self.B @ self.L)) @ self.B)

        return self.lr

    def set_sampling_time(self, dt):
        """
        Set sampling time.

        :param dt: system sampling time
        :type dt: float
        """
        self.dt = dt

    def set_reference(self, ref):
        """
        Set reference for controller.

        :param ref: 2x1 vector
        :type ref: np.ndarray
        """
        self.ref = ref

    def update_integral(self, x):
        """
        Update the integral term for integral action.

        :param x: state
        :type x: np.ndarray 2x1
        """
        if self.dt is None:
            print("[controller] System sampling time not set.\n \
                  Set dt with 'set_sampling_time' method.")
        
        # todo: complete integral action and remove error
        raise NotImplementedError("Integral action update not implemented yet")
        
        self.i_term += # fill in the integral term 

    def reset_integral(self):
        """
        Reset the integral action of the controller
        """
        self.i_term = 0.0

    def set_integral_gain(self, ki):
        """
        Set integral action gain.

        :param ki: integral gain
        :type ki: float
        """
        self.Ki = ki

    def activate_integral_action(self, dt, ki):
        """
        Helper method to activate integral control law

        :param dt: system sampling time
        :type dt: float
        :param ki: integral gain
        :type ki: float
        """
        self.use_integral = True
        self.set_sampling_time(dt)
        self.set_integral_gain(ki)
        self.reset_integral()

    def control_law(self, x):
        """
        Nonlinear control law.

        :param x: state
        :type x: np.ndarray
        :param ref: cart reference position, defaults to 10
        :type ref: float, optional
        """

        if self.use_integral is True:
            self.update_integral(x)
        
        # todo: complete control law and remove error
        raise NotImplementedError("Control law not implemented yet")
    
        u = # fill in the control law
        return u