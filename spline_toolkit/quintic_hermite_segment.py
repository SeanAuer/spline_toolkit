import numpy as np

class QuinticHermiteSegment:
    """
    Class for Quintic Hermite Spline interpolation."
    """
    def __init__(self, points, tangents, curvatures):
        """
        Initialize a Quintic Hermite Segment using grouped position, velocity, and acceleration.
        :param points: [p0, p1]
        :param tangents: [v0, v1]
        :param curvatures: [a0, a1]
        All entries must be array-like of the same shape (1D, 2D, or 3D).
        """
        self.p0, self.p1 = np.array(points[0]), np.array(points[1])
        self.v0, self.v1 = np.array(tangents[0]), np.array(tangents[1])
        self.a0, self.a1 = np.array(curvatures[0]), np.array(curvatures[1])

        if not (self.p0.shape == self.p1.shape == self.v0.shape == self.v1.shape == self.a0.shape == self.a1.shape):
            raise ValueError("All inputs must have the same shape (1D, 2D, or 3D).")

        self.dim = self.p0.shape[0]

        self.coefficients = np.stack([
            self._compute_coeffs_1d(self.p0[d], self.p1[d],
                                    self.v0[d], self.v1[d],
                                    self.a0[d], self.a1[d])
            for d in range(self.dim)
        ], axis=0)

    def _compute_coeffs_1d(self, p0, p1, v0, v1, a0, a1):
        """
        Solve for quintic polynomial coefficients in 1D.
        Returns: np.array([c0, c1, c2, c3, c4, c5])
        """
        A = np.array([
            [1, 0, 0,    0,     0,     0],
            [0, 1, 0,    0,     0,     0],
            [0, 0, 1,    0,     0,     0],
            [1, 1, 1,    1,     1,     1],
            [0, 1, 2,    3,     4,     5],
            [0, 0, 2,    6,    12,    20],
        ])
        b = np.array([p0, v0, a0, p1, v1, a1])
        return np.linalg.solve(A, b)

    def evaluate(self, t):
        """
        Evaluate the segment at parameter t in [0, 1].
        """
        T = np.array([1, t, t**2, t**3, t**4, t**5])
        return self.coefficients @ T

    def _curvature_magnitude(self, t):
        """
        Estimate the curvature magnitude at parameter t.
        For 2D/3D: ||x'(t) × x''(t)|| / ||x'(t)||^3
        For 1D: curvature is zero.
        """
        # First and second derivatives
        T1 = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        T2 = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        d1 = self.coefficients @ T1
        d2 = self.coefficients @ T2
        if self.dim == 1:
            return 0.0
        elif self.dim == 2:
            # Cross product in 2D is scalar
            num = np.abs(d1[0]*d2[1] - d1[1]*d2[0])
            denom = np.linalg.norm(d1)**3 + 1e-8
            return num / denom
        else:
            num = np.linalg.norm(np.cross(d1, d2))
            denom = np.linalg.norm(d1)**3 + 1e-8
            return num / denom

    def sample(self, n_points=100):
        """
        Generate sampled points along the segment, biased toward regions of high curvature.
        """
        # 1. Uniformly sample t at high resolution
        t_hr = np.linspace(0, 1, 500)
        curvatures = np.array([self._curvature_magnitude(t) for t in t_hr])
        # Add a small baseline so straight regions still get sampled
        baseline = 0.01 * np.max(curvatures) if np.max(curvatures) > 0 else 1.0
        weights = curvatures + baseline
        # 2. Compute CDF
        cdf = np.cumsum(weights)
        cdf = cdf / cdf[-1]
        # 3. Inverse transform sampling
        u = np.linspace(0, 1, n_points)
        t_samples = np.interp(u, cdf, t_hr)
        self.sampled_points = np.array([self.evaluate(t) for t in t_samples])
        return self.sampled_points


    def report(self):
        print("Quintic Hermite Segment Parameters:")
        print(f"p0: {self.p0}, v0: {self.v0}, a0: {self.a0}")
        print(f"p1: {self.p1}, v1: {self.v1}, a1: {self.a1}")