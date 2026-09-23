import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



class RaceTrack :
    def __init__(self):
        
        self.width = 5.0
        self.centerline, self.heading = self.generate_track()
        
        self.s           = np.linspace(0, 1, len(self.centerline))
        self.ds          = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        self.curvature   = np.diff(self.heading) / self.ds
        self.curvature   = np.hstack((self.curvature, self.curvature[-1]))
        self.road_length = np.sum(self.ds)



    def position(self,s):
        """Get position on the centerline given a normalized arc-length s in [0, 1]."""
        s = np.clip(s, 0, 1)

        x = np.interp(s, self.s, self.centerline[:,0])
        y = np.interp(s, self.s, self.centerline[:,1])

        return np.array([x, y]) 

    def curvature(self,s):
        """Get curvature on the centerline given a normalized arc-length s in [0, 1]."""
        s = np.clip(s, 0, 1)

        k = np.interp(s, self.s, self.curvature)

        return k

    def _heading(self,s):
        """Get heading on the centerline given a normalized arc-length s in [0, 1]."""
        s = np.clip(s, 0, 1)

        h = np.interp(s, self.s, self.heading)

        return h
    
    def normal(self,s):
        """Get normal vector on the centerline given a normalized arc-length s in [0, 1]."""
        s = np.clip(s, 0, 1)

        h = self._heading(s)

        n = np.array([-np.sin(h), np.cos(h)])

        return n

    def generate_arc(self,start, heading, direction = "left",radius = 10, spanned_angle = np.pi/2, num_points=500):
            """Generate points along a circular arc."""
            
            theta = np.linspace(0, spanned_angle, num_points)
            
            dd = -1 if direction == "left" else 1

            x     =    radius * (np.sin(theta))
            y     = dd*radius * (np.cos(theta)) - dd*radius

            # rotate coordinates to the heading
            x_rot = x * np.cos(heading) - y * np.sin(heading)
            y_rot = x * np.sin(heading) + y * np.cos(heading)

            # translate to start
            x_final = x_rot + start[0]
            y_final = y_rot + start[1]

            heading = heading - dd*theta
            
            
            return np.column_stack((x_final, y_final)), heading

    def generate_straight(self,start, heading , length , num_points=500):
        """Generate points along a straight line segment."""
        
        length = length
        x = np.linspace(0, length, num_points)
        y = np.zeros(num_points)
        # rotate coordinates to the heading
        x_rot = x * np.cos(heading) - y * np.sin(heading)
        y_rot = x * np.sin(heading) + y * np.cos(heading)
        # translate to start
        x_final = x_rot + start[0]
        y_final = y_rot + start[1]

        heading = heading*np.ones(num_points)


        return np.column_stack((x_final, y_final)), heading
    
    def generate_track(self):
        """Generate a closed-loop track with straight segments + arcs."""
        points  = []
        heading = []

        # Example: rectangle with rounded corners
        n_points = 300

        # Bottom straight
        arc1,heading1 = self.generate_straight(start = np.array([0,0]), heading = 0, length = 10, num_points = n_points)

        points.append(arc1)
        heading.append(heading1)


        # Right turn (quarter circle)
        arc2,heading2 = self.generate_arc(arc1[-1,:], heading = heading1[-1], radius = 10, spanned_angle= np.pi/2, num_points = n_points, direction = "right")
        
        points.append(arc2[1:,:])
        heading.append(heading2[1:])

        arc3,heading3 = self.generate_arc(arc2[-1,:], heading = heading2[-1], radius = 20, spanned_angle= np.pi, num_points = n_points, direction = "left")
        
        points.append(arc3[1:,:])
        heading.append(heading3[1:])

        arc4,heading4 = self.generate_arc(arc3[-1,:], heading = heading3[-1], radius = 10, spanned_angle= np.pi/2, num_points = n_points, direction = "right")
        points.append(arc4[1:,:])
        heading.append(heading4[1:])

        # turn right
        arc5,heading5 = self.generate_arc(arc4[-1,:], heading = heading4[-1], radius = 10, spanned_angle= np.pi/2, num_points = n_points, direction = "right")
        points.append(arc5[1:,:])
        heading.append(heading5[1:])

        # straight 
        arc6,heading6 = self.generate_straight(start = arc5[-1,:], heading = heading5[-1], length = 50, num_points = n_points)
        points.append(arc6[1:,:])
        heading.append(heading6[1:])
        
        # right
        arc7,heading7 = self.generate_arc(arc6[-1,:], heading = heading6[-1], radius = 10, spanned_angle= np.pi/2, num_points = n_points, direction = "right")
        points.append(arc7[1:,:])
        heading.append(heading7[1:])

        # straight
        arc8,heading8 = self.generate_straight(start = arc7[-1,:], heading = heading7[-1], length = 40, num_points = n_points)
        points.append(arc8[1:,:])
        heading.append(heading8[1:])

        # right of 30 degrees
        arc9,heading9 = self.generate_arc(arc8[-1,:], heading = heading8[-1], radius = 10, spanned_angle= np.pi/4, num_points = n_points, direction = "right")
        points.append(arc9[1:,:])
        heading.append(heading9[1:])

        # straight
        arc10,heading10 = self.generate_straight(start = arc9[-1,:], heading = heading9[-1], length = 70.7, num_points = n_points)
        points.append(arc10[1:,:])
        heading.append(heading10[1:])

        # right turn
        arc11,heading11 = self.generate_arc(arc10[-1,:], heading = heading10[-1], radius = 10, spanned_angle= 3/4*np.pi, num_points = n_points, direction = "right")
        points.append(arc11[1:,:])
        heading.append(heading11[1:])

        # straight
        arc12,heading12 = self.generate_straight(start = arc11[-1,:], heading = heading11[-1], length = 20, num_points = n_points)
        points.append(arc12[1:,:])
        heading.append(heading12[1:])

        # Concatenate all segments
        centerline = np.vstack(points)
        heading    = np.hstack(heading)

        return centerline, heading

    def offset_curve(self,curve, offset):
        """Offset a curve left/right by given distance."""
        x, y = curve[:,0], curve[:,1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        length = np.sqrt(dx**2 + dy**2)
        nx, ny = -dy/length, dx/length  # normals
        return np.column_stack((x + offset*nx, y + offset*ny))

    def plot_track(self):
        left_boundary = self.offset_curve(self.centerline, self.width/2)
        right_boundary = self.offset_curve(self.centerline, -self.width/2)

        # Build polygon patch
        road = np.vstack([left_boundary, right_boundary[::-1]])

        fig, ax = plt.subplots(figsize=(6,6))
        ax.add_patch(Polygon(road, closed=True, color="lightgray"))
        ax.plot(self.centerline[:,0], self.centerline[:,1], 'k--', linewidth=1, label="Centerline")
        ax.set_aspect('equal')
        ax.set_title("Racetrack Example")
        ax.legend()

        # plot obstacle patch
        import matplotlib.patches as patches
        obstacle = patches.Rectangle((80, -50), 2.5, 25, linewidth=1, edgecolor='r', facecolor='red')
        ## add text obstacle in vertical center of rectangle
        ax.text(80+1.25, -50+12.5, 'Obstacle', color='black', fontsize=8, ha='center', va='center', rotation=90)
        ax.add_patch(obstacle)

        return ax

if __name__ == "__main__":
    
    RaceTrack().plot_track()
    plt.show()