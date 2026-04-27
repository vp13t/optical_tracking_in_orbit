import numpy as np
import matplotlib.pyplot as plt

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.objects as mt_obj
import estimators.eif as eif
import estimators.tune as tune
import visualization.animation_3d as video
import visualization.cw_plot as cw_plot
import visualization.err_plot as err_plot
import sim

if __name__ == "__main__":
    dt = 1
    # SS:MM:HH
    duration = 60 * 30 * 1
    tplot, xhist_sat1, xhist_sat2, measurements = sim.sim(42, dt, duration)

    x0_sat1 = xhist_sat1[:,0]
    x0_sat2 = xhist_sat2[:,0]
    x0_est = sim.init_est(xhist_sat1)
    # x0_est = x0_sat2 - x0_sat1

    I0 = np.linalg.inv(tune.P0)
    estimator = eif.EIF(x0_est, I0, tune.Q, tune.R)

    target_obj = mt_obj.Sat(x0_sat2[:3], area=10, reflectivity=0.9)
    estimated_target_obj = mt_obj.Sat(x0_est[:3], area=10, reflectivity=0.9)

    est_xhist = [x0_est + x0_sat1]
    est_Phist = [np.linalg.inv(I0)]

    for i in range(xhist_sat1.shape[1]-1):
        curr_state = xhist_sat1[:,i]
        next_state = xhist_sat1[:,i+1]
        theta = (np.deg2rad(15) * i) % np.deg2rad(360)

        meas, theta = measurements[i]

        f, F = dn.rel_dyn(curr_state)
        xhat, Pk = estimator.prediction(dt, f, F)
        if meas is not None:
            # estimated_target_obj.update_pos(xhat[:3])
            for y in meas:
                h, H = mt.gen_h_rel(next_state, theta, estimated_target_obj)
                xhat, Pk = estimator.measurement(y, h, H)

        # estimated_target_obj.update_pos(xhat[:3])
        est_xhist.append(xhat + next_state)
        est_Phist.append(Pk)

    est_xhist = np.array(est_xhist).T

    r0 = dn_c.earth_rad + 300000
    video.plot(xhist_sat1, xhist_sat2, est_xhist, measurements, r0, debug=False)
    err_plot.plot(tplot, xhist_sat2, est_xhist, est_Phist)
    cw_plot.plot(tplot, xhist_sat1, xhist_sat2, est_xhist)
    plt.show()
