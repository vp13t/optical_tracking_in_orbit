import numpy as np

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.objects as mt_obj

def sim(seed, dt, duration):
    rng = np.random.default_rng(seed=seed)

    r0 = dn_c.earth_rad + 300000
    v_circ = np.sqrt(dn_c.G * dn_c.earth_mass / r0)

    sat2_z0 = r0 + rng.normal(500, 1000/2)
    sat2_y0 = rng.normal(0, 5000/2)
    sat2_x0 = rng.normal(5000, 5000/2)
    sat2_r0_approx = np.array([sat2_x0, sat2_y0, sat2_z0])
    sat2_r0_n = sat2_r0_approx / np.linalg.norm(sat2_r0_approx)
    sat2_r0 = sat2_z0 * sat2_r0_n
    sat2_v_circ = np.sqrt(dn_c.G * dn_c.earth_mass / sat2_z0)
    sat2_v0 = v_circ * np.cross(sat2_r0_n, -np.array([0,1,0]))

    x0_sat1 = np.array([0, 0, r0, v_circ, 0, 0])
    x0_sat2 = np.concatenate((sat2_r0, sat2_v0))

    tplot, xhist_sat1 = dn.propagate_dyn(x0_sat1, dt, duration)
    _, xhist_sat2 = dn.propagate_dyn(x0_sat2, dt, duration, Q=dn_c.true_Q)

    target_obj = mt_obj.Sat(x0_sat2[:3], area=10, reflectivity=0.9)

    measurements = []
    for i in range(xhist_sat1.shape[1]-1):
        theta = (np.deg2rad(15) * i) % np.deg2rad(360)

        target_obj.update_pos(xhist_sat2[:3,i+1])
        im = mt.gen_camera_image(xhist_sat1[:,i+1], theta, [target_obj])
        meas = mt.meas_from_camera_image(im)
        measurements.append((meas, theta))

    return tplot, xhist_sat1, xhist_sat2, measurements

def init_est(xhist_sat1):
    x0_sat1 = xhist_sat1[:,0]
    r0 = dn_c.earth_rad + 300000
    v_circ = np.sqrt(dn_c.G * dn_c.earth_mass / r0)
    x0_est = dn.f(x0_sat1, dt=5000/v_circ) - x0_sat1
    return x0_est
