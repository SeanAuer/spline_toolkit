import math
import numpy as np
import matplotlib.pyplot as plt
from spline_toolkit import QuinticHermiteSegment

# Define start and end points
p0 = np.array([0.0, 0.0])
p1 = np.array([1.0, 0.0])

# Define tangents as 2D vectors
t0 = np.array([0.0, 1.0])   # Upward direction
t1 = np.array([np.cos(math.radians(-8)), np.sin(math.radians(-8))])  # Slight downward right

# Define curvatures
c0 = np.array([15.0, -4.87])
c1 = np.array([0.0, 0.0])

segment = QuinticHermiteSegment(points=[p0, p1], tangents=[t0, t1], curvatures=[c0, c1])

# Sample points from the segment
points = segment.sample(n_points=1000)
x_vals, y_vals = points[:, 0], points[:, 1]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='Quintic Hermite Segment', color ='blue')
plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'ro', label='Endpoints')
plt.quiver(p0[0], p0[1], t0[0], t0[1], angles='xy', scale_units='xy', scale=1, color='g', label='Tangent Start')
plt.quiver(p1[0], p1[1], t1[0], t1[1], angles='xy', scale_units='xy', scale=1, color='m', label='Tangent End')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Quintic Hermite Segment Example - Airfoil-like Shape')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-0.5, 2.0])
plt.ylim([-1.5, 2.5])
plt.tight_layout()
plt.show()