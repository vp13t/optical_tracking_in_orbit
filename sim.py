import numpy as np

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.objects as mt_obj
import measurement.stars as stars

def sim(seed, dt, duration, use_stars=False):
    rng = np.random.default_rng(seed=seed)

    r0 = dn_c.earth_rad + 300000
    v_circ = np.sqrt(dn_c.G * dn_c.earth_mass / r0)

    sat2_z0 = r0 + rng.normal(500, 1000/2)
    sat2_y0 = rng.normal(5000, 10000/2)
    sat2_x0 = rng.normal(10000, 10000/2)
    sat2_r0_approx = np.array([sat2_x0, sat2_y0, sat2_z0])
    sat2_r0_n = sat2_r0_approx / np.linalg.norm(sat2_r0_approx)
    sat2_r0 = sat2_z0 * sat2_r0_n
    sat2_v_circ = np.sqrt(dn_c.G * dn_c.earth_mass / sat2_z0)
    sat2_v0 = v_circ * np.cross(np.array([0,1,0]), sat2_r0_n)

    x0_sat1 = np.array([0, 0, r0, v_circ, 0, 0])
    x0_sat2 = np.concatenate((sat2_r0, sat2_v0))

    tplot, xhist_sat1 = dn.propagate_dyn(x0_sat1, dt, duration, rng)
    _, xhist_sat2 = dn.propagate_dyn(x0_sat2, dt, duration, rng, Q=dn_c.true_Q)

    target_obj = mt_obj.Sat(x0_sat2[:3], area=10, reflectivity=0.9)
    bright_objs = [target_obj]
    if use_stars:
        bright_objs.extend(stars.STARS)

    rot_per_s = np.deg2rad(15)

    measurements = []
    for i in range(xhist_sat1.shape[1]-1):
        expected_theta = (rot_per_s * i) % np.deg2rad(360)
        # sigma = 10 ms = 1/100 s
        true_theta = expected_theta + rng.normal(0,rot_per_s/100)


        target_obj.update_pos(xhist_sat2[:3,i+1])
        im = mt.gen_camera_image(xhist_sat1[:,i+1], true_theta, bright_objs, rng)
        meas = mt.meas_from_camera_image(im)

        measurements.append((meas, expected_theta))

    return tplot, xhist_sat1, xhist_sat2, measurements

def init_est(xhist_sat1):
    x0_sat1 = xhist_sat1[:,0]
    r0 = dn_c.earth_rad + 300000
    v_circ = np.sqrt(dn_c.G * dn_c.earth_mass / r0)
    x0_est = dn.f(x0_sat1, dt=5000/v_circ) - x0_sat1
    return x0_est
