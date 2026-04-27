import numpy as np
import matplotlib.pyplot as plt

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.rotation as rot
import measurement.objects as mt_obj
import measurement.neg_info as mt_ni
import estimators.gmf as gmf
import estimators.hkf as hkf
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
    duration = 60 * 5 * 1
    tplot, xhist_sat1, xhist_sat2, measurements = sim.sim(10, dt, duration)
    x0_sat1 = xhist_sat1[:,0]
    x0_est = sim.init_est(xhist_sat1)

    S = np.linalg.cholesky(tune.P0)
    x0_est_sp = [x0_est]
    lam = 0.75**2 * (6 + 0) - 6
    for i in range(6):
        offset = np.sqrt(6 + lam) * S[:,i]
        x0_est_sp.append(x0_est + offset)
        x0_est_sp.append(x0_est - offset)

    estimators = [hkf.HKF(x0i, tune.P0/2, tune.Q, tune.R) for x0i in x0_est_sp]
    gmf_estimator = gmf.GMF(estimators, max_filter_count=8)

    estimated_target_obj = mt_obj.Sat(x0_est[:3], area=10, reflectivity=0.9)

    map_est_xhist = [gmf_estimator.map_estimate()[0] + x0_sat1]
    map_est_Phist = [gmf_estimator.map_estimate()[1]]
    est_xhists = [[filter.x + x0_sat1 for filter in gmf_estimator.filters]]
    est_Phists = [[filter.P for filter in gmf_estimator.filters]]
    weights_hist = [gmf_estimator.weights]

    for i in range(xhist_sat1.shape[1]-1):
        curr_state = xhist_sat1[:,i]
        next_state = xhist_sat1[:,i+1]

        meas, theta = measurements[i]

        f, F = dn.rel_dyn(curr_state)
        xhats, Pks = gmf_estimator.prediction(dt, f, F)
        if meas is not None:
            for y in meas:
                h, _ = mt.gen_h_rel(next_state, theta, estimated_target_obj)
                xhats, Pks = gmf_estimator.measurement(y, h, tune.R)
        else:
            h = mt_ni.gen_h_rel(next_state, theta, estimated_target_obj)
            xhats, Pks = gmf_estimator.measurement(-np.ones(1), h, tune.R_ni, reweight_only=True)

        est_xhists.append([next_state + xhat for xhat in xhats])
        est_Phists.append(Pks)
        weights_hist.append(gmf_estimator.weights)

        map_est_xhist.append(next_state + gmf_estimator.map_estimate()[0])
        map_est_Phist.append(gmf_estimator.map_estimate()[1])

    map_est_xhist = np.array(map_est_xhist).T
    weights_hist = np.array(weights_hist)

    r0 = r0 = dn_c.earth_rad + 300000
    # video.plot(xhist_sat1, xhist_sat2, map_est_xhist, r0, debug=False)
    err_plot.plot(tplot[1:], xhist_sat2[:,1:], map_est_xhist[:,1:], map_est_Phist[1:])
    cw_plot.plot_sat1frame(tplot, xhist_sat1, xhist_sat2, map_est_xhist)
    prob_heatmap.plot(xhist_sat1, xhist_sat2, est_xhists, est_Phists, weights_hist)

    plt.show()
