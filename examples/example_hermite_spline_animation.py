import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from spline_toolkit import QuinticHermiteSpline, QuinticHermiteSegment

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
    angle0 = 0 - 120 * np.sin(frame * 0.1)
    angle1 = 180 - 90 * np.sin(frame * 0.1)
    t0 = np.array([np.cos(math.radians(angle0)), np.sin(math.radians(angle0))])
    t1 = np.array([np.cos(math.radians(angle1)), np.sin(math.radians(angle1))])

    segment = QuinticHermiteSegment(
        [p0, p1],
        [t0, t1],
        [c0, c1]
    )
    spline = QuinticHermiteSpline.from_segments(segment)
    points = spline.sample()
    x_vals, y_vals = points[:, 0], points[:, 1]

    line.set_data(x_vals, y_vals)

    # Update quivers
    arrow1.set_offsets(p0)
    arrow1.set_UVC(t0[0], t0[1])
    arrow2.set_offsets(p1)
    arrow2.set_UVC(t1[0], t1[1])

    return line, arrow1, arrow2, point_plot

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=50)
plt.title('Animated Quintic Hermite Spline - Tangency sweep')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()