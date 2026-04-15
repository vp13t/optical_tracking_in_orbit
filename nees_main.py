import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import scipy

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.rotation as rot
import measurement.objects as mt_obj
import measurement.neg_info as mt_ni
import estimators.ukf as ukf
import estimators.hkf as hkf
import visualization.animation_3d as video
import visualization.cw_plot as cw_plot
import visualization.err_plot as err_plot
import sim

def run(args):
    seed, dt, duration = args
    tplot, xhist_sat1, xhist_sat2, measurements = sim.sim(seed, dt, duration)
    x0_est = sim.init_est(xhist_sat1)

    m = "hkf"
    if m == "hkf":
        Q = 100.0 * (np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]) ** 2)
        R = 1.0 * (np.diag([50.0, 50.0, 1.0]) ** 2)
        R_ni = 2.0 * (np.diag([2.0]) ** 2)
        P0 = 1.0 * (np.diag([5000.0, 2000.0, 5000.0, 1000.0, 1000.0, 1000.0]) ** 2)
        estimator = hkf.HKF(x0_est, P0, Q, R)
    elif m == "ukf":
        Q = 10.0 * (np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]) ** 2)
        R = 10.0 * (np.diag([50.0, 50.0, 5.0]) ** 2)
        R_ni = 1.0 * (np.diag([2.0]) ** 2)
        P0 = 1.0 * (np.diag([5000.0, 2000.0, 5000.0, 1000.0, 1000.0, 1000.0]) ** 2)
        estimator = ukf.UKF(x0_est, P0, Q, R)

    estimated_target_obj = mt_obj.Sat(x0_est[:3], area=10, reflectivity=0.9)

    est_xhist = [x0_est]
    est_Phist = [P0]

    nees = []
    for i in range(xhist_sat1.shape[1]-1):
        curr_state = xhist_sat1[:,i]
        next_state = xhist_sat1[:,i+1]
        target_rel_state = xhist_sat2[:,i+1] - next_state


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
        
        nees.append(estimator.NEES(target_rel_state))
    return np.array(nees)

if __name__ == "__main__":
    dt = 1
    # SS:MM:HH
    duration = 60 * 10 * 1

    results = None
    N = 64
    args = [(seed, dt, duration) for seed in range(N)]
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, args))
    results = np.array(results)

    alpha = 0.05
    lb1 = scipy.stats.chi2.ppf(alpha/2,6)
    ub1 = scipy.stats.chi2.ppf(1-alpha/2,6)

    fig, axs = plt.subplots(1, 2, sharey=True)

    ax = axs[0]
    ax.set_yscale('log')
    ax.axhline(y=6, color='r', linestyle='--')
    ax.axhline(y=lb1, color='b', linestyle='--')
    ax.axhline(y=ub1, color='b', linestyle='--')
    tspan = np.arange(results.shape[1])
    for row in range(results.shape[0]):
        ax.scatter(tspan, results[row,:], s=5)
    ax.set_ylabel("NEES")
    ax.set_xlabel("Time")
    ax.set_title(f"N={N} NEES Simulations")

    lbN = scipy.stats.chi2.ppf(alpha/2,6*N)/N
    ubN = scipy.stats.chi2.ppf(1-alpha/2,6*N)/N

    ax = axs[1]
    ax.set_yscale('log')
    ax.axhline(y=6, color='r', linestyle='--')
    ax.axhline(y=lbN, color='b', linestyle='--')
    ax.axhline(y=ubN, color='b', linestyle='--')
    anees = np.mean(results, axis=0)
    ax.plot(tspan, anees)
    ax.set_ylabel("MNEES")
    ax.set_xlabel("Time")
    ax.set_title("Mean NEES")

    plt.show()
