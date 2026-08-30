import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class EmbeddedSimEnvironment:

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller_class = controller
        self.controller = controller.control_law
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt

        # Sim data
        self.t = self.x_vec = self.u_vec = self.sim_loop_length = None

        # Plotting definitions
        self.plt_window = float("inf")    # running plot window, in seconds, or float("inf")

    def run(self, x0, online_plot=False):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(2, 1)
        u_vec = np.empty((1, 0))

        # Start figure
        if online_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3)
        for i in range(sim_loop_length):
            # Iteration
            print(i, "/", (sim_loop_length - 1))
            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(2, 1)
            u = self.controller(x)
            x_next = self.dynamics(x, u)

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(2, 1), axis=1)
            u_vec = np.append(u_vec, u.reshape(1, 1))

            if online_plot:
                # Get plot window values:
                if self.plt_window != float("inf"):
                    l_wnd = 0 if int(i + 1 - self.plt_window / self.dt) < 1 else int(i + 1 - self.plt_window / self.dt)
                else:
                    l_wnd = 0

                ax1.clear()
                ax1.set_title("Vehicle")
                ax1.plot(t[l_wnd:], x_vec[0, l_wnd:], 'r--')
                ax1.legend(["Y"])
                ax1.set_ylabel("Y [m]")

                ax2.clear()
                ax2.plot(t[l_wnd:], x_vec[1, l_wnd:], 'r--')
                ax2.legend(["vy"])
                ax2.set_ylabel("vy [m/s]")

                ax3.clear()
                ax3.plot(t[l_wnd:-1], u_vec[l_wnd:], 'r--')
                ax3.legend(["u"])
                ax3.set_ylabel("dteering rate [rad/s]")

                plt.pause(0.001)

        if online_plot:
            plt.show()

        # Store data internally for offline plotting
        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.sim_loop_length = sim_loop_length

        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3) = plt.subplots(3)


        # Add the patch to the axes
       


        
        ax1.clear()
        ax1.set_title("Vehicle")
        ax1.plot(t, x_vec[0, :], 'r--')
        ax1.axhline(self.controller_class.ref[0], color='g',linestyle='--')
        ax1.axhline(self.controller_class.ref[0]*0.5, color='b',linestyle='--', label = "(75% of ref)")
        ax1.legend()
        ax1.set_ylabel("Y [m]")
        ax1.set_ylim(-1, 2)

        ax2.clear()
        ax2.plot(t, x_vec[1, :], 'r--')
        ax2.axhline(self.controller_class.ref[1], color='g', linestyle='--')
        ax2.legend()
        ax2.set_ylabel("vy [m/s]")

        ax3.clear()
        ax3.plot(t[:-1], u_vec, 'r--')
        ax3.legend()
        ax3.set_ylabel(r"$u$ [rad/s]")
        ax3.set_xlabel(r"time [s]")



        # Define road parameters
        road_x, road_y = 0, -2.
        road_width  = self.total_sim_time*self.model.v_ref   # in plot units
        road_height = 4  # in plot units

        # Create a gray rectangle patch
        road_patch = patches.Rectangle(
            (road_x, road_y),     # bottom-left corner
            road_width,           # width
            road_height,          # height
            facecolor='gray',     # fill color
            edgecolor='black'     # outline color
        )
        
        obstcle_width = 1.5
        obstacle_pos = 200  # position of the obstacle in meters
        obstacle_patch = patches.Rectangle(
            (obstacle_pos, -1- obstcle_width/2 ),  # bottom-left corner
            100 ,                 # width
            obstcle_width ,                 # height
            facecolor='red',   # fill color
            edgecolor='black'  # outline color
        )

        fig,ax = plt.subplots(figsize = (20, 5))
        ax.add_patch(road_patch)
        ax.add_patch(obstacle_patch)
        ax.axhline(road_y + road_height/2, color='white', linewidth=4, linestyle='--')  # center line
        x      = t*self.model.v_ref
        y      = x_vec[0, :]
        ax.plot(x, y, 'r--')
        ax.set_title("Vehicle trajectory")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Y [m]")

        ax.set_ylim((road_y)*1.2, (road_y + road_height)*1.2)
        ax.set_xlim(0, road_width)
        

        plt.show()

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window