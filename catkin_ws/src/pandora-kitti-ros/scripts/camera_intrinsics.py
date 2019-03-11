"""
The intrinsic parameters for Pandora front facing color camera.
You can verify these parameters by looking in the Pandora Users' manual, page 7
"""
import numpy as np
from math import tan, pi
"""
The intrinsic matrix is defined as

| f 0 Cu |
| 0 f Cv |
| 0 0 1  |
where f is the focal length in pixels, and (Cu, Cv) is the central point of the image
in pixel coordinates
"""
CAMERA_RESOLUTION_X, CAMERA_RESOLUTION_Y = (1280, 720)
FOCAL_LENGTH_IN_MM = 5.47
HORIZONTAL_FOV_IN_DEGREES = 52
APPROX_FOCAL_LENGTH_IN_PIXELS = (
    CAMERA_RESOLUTION_X // 2) / tan(HORIZONTAL_FOV_IN_DEGREES * 0.5 * pi/180)

K = np.identity(3)
K[0, 2] = CAMERA_RESOLUTION_X // 2
K[1, 2] = CAMERA_RESOLUTION_Y // 2
K[0, 0] = APPROX_FOCAL_LENGTH_IN_PIXELS
K[1, 1] = APPROX_FOCAL_LENGTH_IN_PIXELS
K[2, 2] = 1
