import math
import numpy as np
import matplotlib.pyplot as plt
from spline_toolkit import QuinticHermiteSegment, QuinticHermiteSpline

# Define start and end points
p0 = np.array([0.0, 0.0])
p1 = np.array([1.0, 0.0])
p2 = np.array([2.0, 0.0])  
ctrl_pts = np.array([p0, p1, p2])

# Define tangents
t0 = [np.cos(math.radians(90.0)), np.sin(math.radians(90.0))]  
t1 = [np.cos(math.radians(-8.0)), np.sin(math.radians(-8.0))]
t2 = [np.cos(math.radians(90.0)), np.sin(math.radians(90.0))]
ctrl_tgts = np.array([t0, t1, t2])

# Define curvatures
c0 = np.array([15.0, -4.87])
c1 = np.array([0.0, 0.0])
c2 = np.array([-4.87, 7.0])
ctrl_crvs = np.array([c0, c1, c2])

curve = QuinticHermiteSpline.from_controls(ctrl_pts, ctrl_tgts, ctrl_crvs)

# Use pre-computed samples from spline with denser, curvature-weighted sampling
points = curve.sample(n_points=300)
x_vals, y_vals = points[:, 0], points[:, 1]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='Quintic Hermite Spline', color ='blue')
plt.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]],'ro', label='Control Points')
plt.quiver(p0[0], p0[1], t0[0], t0[1], angles='xy', scale_units='xy', scale=1, color='g', label='Tangent Start')
plt.quiver(p1[0], p1[1], t1[0], t1[1], angles='xy', scale_units='xy', scale=1, color='m', label='Tangent Mid')
plt.quiver(p2[0], p2[1], t2[0], t2[1], angles='xy', scale_units='xy', scale=1, color='m', label='Tangent End')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Quintic Hermite Spline Example - Adding a Point')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-0.5, 3.0])
plt.ylim([-1.5, 2.5])
plt.tight_layout()
plt.show()
