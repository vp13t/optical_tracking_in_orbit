import numpy as np
import matplotlib.pyplot as plt

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.rotation as rot
import measurement.objects as mt_obj
import measurement.neg_info as mt_ni
import estimators.pdaf as pdaf
import estimators.tune as tune
import visualization.animation_3d as video
import visualization.cw_plot as cw_plot
import visualization.err_plot as err_plot
import visualization.probability_mass_plot as prob_heatmap
import sim

if __name__ == "__main__":
    # x0_est= x0_sat2 - x0_sat1
    dt = 1
    # SS:MM:HH
    duration = 60 * 30 * 1
    tplot, xhist_sat1, xhist_sat2, measurements = sim.sim(1, dt, duration, use_stars=True)
    x0_sat1 = xhist_sat1[:,0]
    x0_est = sim.init_est(xhist_sat1)

    PG = 0.996
    PD = 0.9
    Lambda = 10.0
    estimator = pdaf.PDAF(x0_est, tune.P0, tune.Q, tune.R, PG, PD, Lambda)

    estimated_target_obj = mt_obj.Sat(x0_est[:3], area=10, reflectivity=0.9)

    est_xhist = [x0_est + x0_sat1]
    est_Phist = [tune.P0]

    for i in range(xhist_sat1.shape[1]-1):
        curr_state = xhist_sat1[:,i]
        next_state = xhist_sat1[:,i+1]

        meas, theta = measurements[i]

        f, F = dn.rel_dyn(curr_state)
        xhat, Pk = estimator.prediction(dt, f, F)
        if meas is not None:
            h, _ = mt.gen_h_rel(next_state, theta, estimated_target_obj)
            filtered_ys, validated_likelihoods = estimator.pda(meas, h, tune.R)
            for y in filtered_ys:
                xhat, Pk = estimator.measurement(y, h, tune.R)
        # else:
        #     h = mt_ni.gen_h_rel(next_state, theta, estimated_target_obj)
        #     xhat, Pk = estimator.measurement(-np.ones(1), h, tune.R_ni)

        # estimated_target_obj.update_pos(xhat[:3])
        est_xhist.append(next_state + xhat)
        est_Phist.append(Pk)

    est_xhist = np.array(est_xhist).T

    r0 = r0 = dn_c.earth_rad + 300000
    video.plot(xhist_sat1, xhist_sat2, est_xhist, r0, debug=False, plot_stars=True)
    err_plot.plot(tplot[1:], xhist_sat2[:,1:], est_xhist[:,1:], est_Phist[1:])
    cw_plot.plot_sat1frame(tplot, xhist_sat1, xhist_sat2, est_xhist)

    plt.show()
