# spline_toolkit

`spline_toolkit` is a lightweight Python library for constructing and visualizing parametric splines, with an initial implementation of 2D quintic Hermite splines and spline curves. The toolkit is designed for applications in geometry definition, aerodynamic surface generation, and CAD/CAE workflows.

## Features

- Quintic Hermite spline definition with control of position, tangent (as angle), and curvature (as vectors)
- Continuous spline curves made of multiple Hermite splines
- Automatic sampling and plotting support via NumPy and Matplotlib
- Modular design to support additional spline types in future releases

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

## Example Usage

```python
from spline_toolkit import QuinticHermiteSpline

import numpy as np
import math

p0 = np.array([0.0, 0.0])
p1 = np.array([1.0, 0.0])
t0 = math.radians(90)
t1 = math.radians(-10)
c0 = np.array([5.0, -5.0])
c1 = np.array([0.0, 0.0])

spline = QuinticHermiteSpline(points=[p0, p1], tangents=[t0, t1], curvatures=[c0, c1])
```

## License

Licensed under the Apache License 2.0. See `LICENSE` file for details.

## Planned Extensions

- B-spline and Bézier curve implementations
- 3-D spline and curve support
