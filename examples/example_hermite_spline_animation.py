import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from spline_toolkit import QuinticHermiteSpline

# Fixed control data
p0 = np.array([0.0, 0.0])
p1 = np.array([1.0, 0.0])
c0 = np.array([0.0, 0.0])
c1 = np.array([0.0, 0.0])

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'b-', label='Spline')
arrow1 = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='g')
arrow2 = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='m')
point_plot, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'ro', label='Endpoints')

def init():
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    return line, arrow1, arrow2, point_plot

def update(frame):
    # Sweep trailing tangent angle and curvature
    t0 = math.radians(0 - 120 * np.sin(frame * 0.1))
    t1 = math.radians(180 - 90 * np.sin(frame * 0.1))

    spline = QuinticHermiteSpline(points=[p0, p1], tangents=[t0, t1], curvatures=[c0, c1])
    points = spline.samples
    x_vals, y_vals = points[:, 0], points[:, 1]

    line.set_data(x_vals, y_vals)

    # Update quivers
    arrow1.set_offsets(p0)
    arrow1.set_UVC(np.cos(t0), np.sin(t0))
    arrow2.set_offsets(p1)
    arrow2.set_UVC(np.cos(t1), np.sin(t1))

    return line, arrow1, arrow2, point_plot

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=50)
plt.title('Animated Quintic Hermite Spline - Tangent and Curvature Sweep')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()