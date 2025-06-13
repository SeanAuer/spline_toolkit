import numpy as np
from spline_toolkit import QuinticHermiteSegment


class QuinticHermiteSpline:
    def __init__(self, segment: QuinticHermiteSegment):
        self.segments = [segment]

    @classmethod
    def from_controls(cls, points, tangents, curvatures):
        if not (len(points) == len(tangents) == len(curvatures)):
            raise ValueError("points, tangents, and curvatures must have the same length")
        obj = cls.__new__(cls)
        obj.segments = []
        for i in range(len(points) - 1):
            seg = QuinticHermiteSegment(
                points=[points[i], points[i + 1]],
                tangents=[tangents[i], tangents[i + 1]],
                curvatures=[curvatures[i], curvatures[i + 1]]
            )
            obj.segments.append(seg)
        return obj

    @classmethod
    def from_segments(cls, *args, tolerance=1e-8):
        # Allow passing either a list of segments or multiple segment arguments
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            segments = list(args[0])
        else:
            segments = list(args)

        if len(segments) < 1:
            raise ValueError("At least one segment must be provided.")

        for i in range(len(segments) - 1):
            s1 = segments[i]
            s2 = segments[i + 1]
            if not (np.allclose(s1.p1, s2.p0, atol=tolerance) and
                    np.allclose(s1.v1, s2.v0, atol=tolerance) and
                    np.allclose(s1.a1, s2.a0, atol=tolerance)):
                raise ValueError(f"Segments {i} and {i+1} are not C² continuous.")

        obj = cls.__new__(cls)
        obj.segments = segments
        return obj

    def add_point(self, point, tangent, curvature, index=None):
        if index is None:
            last = self.segments[-1]
            new_segment = QuinticHermiteSegment(
                points=[last.p1, point],
                tangents=[last.v1, tangent],
                curvatures=[last.a1, curvature]
            )
            self.segments.append(new_segment)
        else:
            if index < 0 or index > len(self.segments):
                raise IndexError("Invalid index for insertion.")
            if index == 0:
                raise ValueError("Cannot insert before the first point.")
            before_seg = self.segments[index - 1]
            new_segment = QuinticHermiteSegment(
                points=[before_seg.p1, point],
                tangents=[before_seg.v1, tangent],
                curvatures=[before_seg.a1, curvature]
            )
            after_seg = QuinticHermiteSegment(
                points=[point, self.segments[index].p1],
                tangents=[tangent, self.segments[index].v1],
                curvatures=[curvature, self.segments[index].a1]
            )
            self.segments = (
                self.segments[:index] + [new_segment, after_seg] + self.segments[index + 1:]
            )

    def sample(self, n_points=None):
        sampled_points = []

        if n_points is None:
            # Default: uniform number of points per segment
            for seg in self.segments:
                sampled_points.extend(seg.sample(n_points=100))
        else:
            # Distribute n_points based on segment curvature
            curvatures = [np.max(np.linalg.norm([seg.a0, seg.a1], axis=1)) for seg in self.segments]
            total_curvature = sum(curvatures)
            if total_curvature == 0:
                weights = [1 / len(self.segments)] * len(self.segments)
            else:
                weights = [c / total_curvature for c in curvatures]

            points_per_seg = [max(2, int(round(w * n_points))) for w in weights]
            for seg, pts in zip(self.segments, points_per_seg):
                sampled_points.extend(seg.sample(n_points=pts))

        return np.array(sampled_points)
