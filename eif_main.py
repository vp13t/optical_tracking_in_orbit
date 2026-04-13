import numpy as np
import matplotlib.pyplot as plt

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.objects as mt_obj
import estimators.eif as eif
import visualization.animation_3d as video
import visualization.cw_plot as cw_plot
import visualization.err_plot as err_plot

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

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
    x0_est = dn.propagate_dyn(x0_sat1, 5000/v_circ, res=100)[1][:,-1] - x0_sat1
    # x0_est = x0_sat2 - x0_sat1

    # Q = np.diag([1, 1, 1, 1, 1, 1]) * 1000000
    Q = np.diag([10, 10, 10, 0.01, 0.01, 0.01])
    R = np.diag([10, 10, 1]) * 10
    I0 = np.linalg.inv(np.diag([5000, 1000, 5000, 10, 10, 10])**2)
    estimator = eif.EIF(x0_est, I0, Q, R)

    target_obj = mt_obj.Sat(x0_sat2[:3], area=10, reflectivity=0.9)
    estimated_target_obj = mt_obj.Sat(x0_est[:3], area=10, reflectivity=0.9)

    dt = 1
    # SS:MM:HH
    duration = 60 * 10 * 1
    tplot, xhist_sat1 = dn.propagate_dyn(x0_sat1, duration, int(duration/dt))
    _, xhist_sat2 = dn.propagate_dyn(x0_sat2, duration, int(duration/dt))

    # cam_state = xhist_sat1[:,0]
    # theta = 0
    # h_test = mt.gen_h(cam_state, theta, estimated_target_obj)
    # print(f"Initial true relative pos: {x0_sat2 - x0_sat1}")
    # print(f"Initial estimated relative pos: {x0_est}")
    # print(f"h(true): {h_test(x0_sat2 - x0_sat1)}")
    # print(f"h(est): {h_test(x0_est)}")
    # exit(0)

    est_xhist = [x0_est+x0_sat1]
    est_Phist = [np.linalg.inv(I0)]

    measurements = []
    for i in range(xhist_sat1.shape[1]-1):
        curr_state = xhist_sat1[:,i]
        next_state = xhist_sat1[:,i+1]
        theta = (np.deg2rad(15) * i) % np.deg2rad(360)

        target_obj.update_pos(xhist_sat2[:3,i+1])
        im = mt.gen_camera_image(xhist_sat1[:,i+1], theta, [target_obj])
        meas = mt.meas_from_camera_image(im)
        measurements.append(meas)

        f, F = dn.rel_dyn(curr_state)
        xhat, Pk = estimator.prediction(dt, f, F)
        if meas is not None:
            # estimated_target_obj.update_pos(xhat[:3])
            for y in meas:
                h = mt.gen_h_rel(next_state, theta, estimated_target_obj)
                H = mt.gen_H_rel(next_state, theta, estimated_target_obj)
                xhat, Pk = estimator.measurement(y, h, H)

        # estimated_target_obj.update_pos(xhat[:3])
        est_xhist.append(xhat + next_state)
        est_Phist.append(Pk)

    est_xhist = np.array(est_xhist).T

    video.plot(xhist_sat1, xhist_sat2, est_xhist, measurements, r0, debug=False)
    err_plot.plot(tplot, xhist_sat2, est_xhist, est_Phist)
    cw_plot.plot(tplot, xhist_sat1, xhist_sat2, est_xhist)
    plt.show()
