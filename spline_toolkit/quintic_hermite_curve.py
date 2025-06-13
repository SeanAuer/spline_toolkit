import numpy as np
from spline_toolkit import QuinticHermiteSpline

class QuinticHermiteCurve:
    def __init__(self, initial_spline: QuinticHermiteSpline, num_samples_per_spline=100):
        self.splines = [initial_spline]
        self.resample(num_samples_per_spline)

    def add_point(self, point, tangent, curvature):
        last = self.splines[-1]
        spline = QuinticHermiteSpline(
            points=[last.p1, point],
            tangents=[last.t1, tangent],
            curvatures=[last.c1, curvature]
        )
        self.splines.append(spline)
        self.resample()

    @classmethod
    def from_controls(cls, points, tangents, curvatures):
        if not (len(points) == len(tangents) == len(curvatures)):
            raise ValueError("points, tangents, and curvatures must have the same length")
        obj = cls.__new__(cls)
        obj.splines = []
        for i in range(len(points) - 1):
            spline = QuinticHermiteSpline(
                points=[points[i], points[i + 1]],
                tangents=[tangents[i], tangents[i + 1]],
                curvatures=[curvatures[i], curvatures[i + 1]]
            )
            obj.splines.append(spline)
        obj.resample()
        return obj

    @classmethod
    def from_splines(cls, splines, tolerance=1e-8):
        for i in range(len(splines) - 1):
            s1 = splines[i]
            s2 = splines[i + 1]
            if not (np.allclose(s1.p1, s2.p0, atol=tolerance) and
                    np.allclose(s1.t1, s2.t0, atol=tolerance) and
                    np.allclose(s1.c1, s2.c0, atol=tolerance)):
                raise ValueError(f"Spline {i} and {i+1} are not C² continuous at their interface.")
        obj = cls.__new__(cls)
        obj.splines = splines
        obj.resample()
        return obj

    def resample(self, num_samples_per_spline=100):
        for spline in self.splines:
            spline.resample(num_samples_per_spline)
        self.samples = np.vstack([spline.samples for spline in self.splines])

    def sample(self, n):
        # Evenly distribute n samples across all splines
        num_splines = len(self.splines)
        samples_per_spline = n // num_splines
        remainder = n % num_splines
        samples = []
        for i, spline in enumerate(self.splines):
            n_samples = samples_per_spline + (1 if i < remainder else 0)
            s = spline.resample(n_samples)
            # Avoid duplicate points at segment boundaries except for the first spline
            if i > 0:
                s = s[1:]
            samples.append(s)
        return np.vstack(samples)