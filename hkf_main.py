import numpy as np
import matplotlib.pyplot as plt

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.rotation as rot
import measurement.objects as mt_obj
import measurement.neg_info as mt_ni
import estimators.hkf as hkf
import visualization.animation_3d as video
import visualization.cw_plot as cw_plot
import visualization.err_plot as err_plot
import sim

if __name__ == "__main__":
    # x0_est= x0_sat2 - x0_sat1
    dt = 1
    # SS:MM:HH
    duration = 60 * 30 * 1
    tplot, xhist_sat1, xhist_sat2, measurements = sim.sim(3, dt, duration)
    x0_sat1 = xhist_sat1[:,0]
    x0_est = sim.init_est(xhist_sat1)

    # Q = 100.0 * (np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]) ** 2)
    # R = 1.0 * (np.diag([50.0, 50.0, 1.0]) ** 2)
    # R_ni = 1.0 * (np.diag([2.0]) ** 2)
    # P0 = 1.0 * (np.diag([10000.0, 2000.0, 10000.0, 1000.0, 1000.0, 1000.0]) ** 2)
    Q = 50.0 * (np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]) ** 2)
    R = 5.0 * (np.diag([100.0, 100.0, 10.0]) ** 2)
    R_ni = 10.0 * (np.diag([2.0]) ** 2)
    P0 = 1.0 * (np.diag([5000.0, 5000.0, 5000.0, 100.0, 100.0, 100.0]) ** 2)
    estimator = hkf.HKF(x0_est, P0, Q, R)

    estimated_target_obj = mt_obj.Sat(x0_est[:3], area=10, reflectivity=0.9)

    est_xhist = [x0_est + x0_sat1]
    est_Phist = [P0]

    for i in range(xhist_sat1.shape[1]-1):
        curr_state = xhist_sat1[:,i]
        next_state = xhist_sat1[:,i+1]

        meas, theta = measurements[i]

        f, F = dn.rel_dyn(curr_state)
        xhat, Pk = estimator.prediction(dt, f, F)
        if meas is not None:
            # estimated_target_obj.update_pos(xhat[:3])
            for y in meas:
                h, _ = mt.gen_h_rel(next_state, theta, estimated_target_obj)
                xhat, Pk = estimator.measurement(y, h, R)
        else:
            h = mt_ni.gen_h_rel(next_state, theta, estimated_target_obj)
            xhat, Pk = estimator.measurement(-np.ones(1), h, R_ni)

        # estimated_target_obj.update_pos(xhat[:3])
        est_xhist.append(next_state + xhat)
        est_Phist.append(Pk)

    est_xhist = np.array(est_xhist).T

    r0 = r0 = dn_c.earth_rad + 300000
    video.plot(xhist_sat1, xhist_sat2, est_xhist, measurements, r0, debug=False)
    err_plot.plot(tplot[1:], xhist_sat2[:,1:], est_xhist[:,1:], est_Phist[1:])
    cw_plot.plot(tplot, xhist_sat1, xhist_sat2, est_xhist)
    plt.show()
