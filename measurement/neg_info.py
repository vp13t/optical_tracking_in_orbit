import numpy as np

from . import measurement as mt
from . import rotation as rot
from . import constants as c
from . import objects as obj

def gen_h_rel(cam_state, theta, sat):
    cam_frame = mt.camera_frame(cam_state, theta)
    Phi = rot.rotation_matrix(rot.InertialFrame(), cam_frame)

    def h(x):
        xc = Phi @ x[:3]
        Xc = xc[0]
        Yc = xc[1]
        Zc = xc[2]

        if np.abs(np.atan2(Xc, Yc)) > c.view_angle/2 or np.abs(np.atan2(Zc, Yc)) > c.view_angle/2:
            return -np.ones(1)
        return np.ones(1)
    return h