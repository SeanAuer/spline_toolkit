import numpy as np

class QuinticHermiteSpline:
    """
    Class for Quintic Hermite Spline interpolation."
    """
    def __init__(self, points, tangents, curvatures=None, num_samples=100):
        """
        Initialize the Quintic Hermite Spline with point and tangency values.
        :param points: List of two points to interpolate.
        :param tangents: List of two tangent values at the specified points.
        :param curvatures: List of two curvature values at the specified points.
        """
        if curvatures is None:
            curvatures = [0, 0]
        if len(points) != 2 or len(tangents) != len(points) or len(curvatures) != len(points):
            raise ValueError(f"Points and tangents must be lists of rank two. \
        )                    Received: {len(points)} points and {len(tangents)} tangent values and {len(curvatures)} curvature values.")
        if type(points) is not np.ndarray:
            points = np.array(points)
        if type(curvatures) is not np.ndarray:
            curvatures = np.array(curvatures)
        self.p0, self.p1 = points
        self.t0, self.t1 = tangents
        # Ensure curvatures are arrays matching p0's shape
        self.c0 = np.array(curvatures[0]) if not isinstance(curvatures[0], np.ndarray) else curvatures[0]
        self.c1 = np.array(curvatures[1]) if not isinstance(curvatures[1], np.ndarray) else curvatures[1]


        def _angle_to_vector(angle, dim):
            # Allow optional magnitude: angle can be float or (angle, magnitude)
            if isinstance(angle, tuple):
                if dim == 2 and len(angle) == 2 and isinstance(angle[0], (float, int)):
                    theta, mag = angle
                elif dim == 3 and len(angle) == 2 and isinstance(angle[0], tuple):
                    (theta, phi), mag = angle
                else:
                    raise ValueError("Invalid angle format.")
            else:
                theta = angle
                mag = 1.0

            if dim == 2:
                v = np.array([np.cos(theta), np.sin(theta)])
            elif dim == 3:
                v = np.array([
                    np.cos(theta) * np.cos(phi),
                    np.sin(theta) * np.cos(phi),
                    np.sin(phi)
                ])
            else:
                raise ValueError("Angle-based input only supported in 2D or 3D.")

            return mag * v

        # Convert angles to vectors if needed
        for i in range(2):
            tangent = self.t0 if i == 0 else self.t1
            if isinstance(tangent, (float, tuple)):
                dim = len(self.p0)
                vec = _angle_to_vector(tangent, dim)
                if i == 0:
                    self.t0 = vec
                else:
                    self.t1 = vec

        # Precompute samples for plotting
        self.resample(num_samples)

    def resample(self, num_samples=100):
        """
        Sample points along the Quintic Hermite Spline.
        :param num_samples: Number of samples to generate.
        :return: Array of sampled points.
        """
        self.samples = np.array([self.evaluate(t) for t in np.linspace(0, 1, num_samples)])
        return self.samples

    def evaluate(self, t):
        """
        Evaluate the Quintic Hermite Spline at a given parameter t.
        :param t: Parameter value in the range [0, 1].
        :return: Interpolated value at parameter t.
        """
        H0 = 1 - 10*t**3 + 15*t**4 - 6*t**5
        H1 = t - 6*t**3 + 8*t**4 - 3*t**5
        H2 = 0.5*t**2 - 1.5*t**3 + 1.5*t**4 - 0.5*t**5
        H3 = 10*t**3 - 15*t**4 + 6*t**5
        H4 = -4*t**3 + 7*t**4 - 3*t**5
        H5 = 0.5*t**3 - t**4 + 0.5*t**5

        return (
            H0 * self.p0 +
            H1 * self.t0 +
            H2 * self.c0 +
            H3 * self.p1 +
            H4 * self.t1 +
            H5 * self.c1
        )

    def report(self):
        print("Quintic Hermite Spline Parameters:")
        print(f"p0: {self.p0}, t0: {self.t0}, c0: {self.c0}")
        print(f"p1: {self.p1}, t1: {self.t1}, c1: {self.c1}")       