import numpy as np
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
import scipy

import dynamics.dynamics as dn
import dynamics.constants as dn_c
import measurement.measurement as mt
import measurement.rotation as rot
import measurement.objects as mt_obj
import measurement.neg_info as mt_ni
import estimators.hkf as hkf
import estimators.gmf as gmf
import estimators.tune as tune
import sim

def run(args):
    seed, dt, duration = args
    tplot, xhist_sat1, xhist_sat2, measurements = sim.sim(seed, dt, duration)
    x0_est = sim.init_est(xhist_sat1)

    S = np.linalg.cholesky(tune.P0)
    x0_est_sp = [x0_est]
    lam = 0.75**2 * (6 + 0) - 6
    for i in range(6):
        offset = np.sqrt(6 + lam) * S[:,i]
        x0_est_sp.append(x0_est + offset)
        x0_est_sp.append(x0_est - offset)

    estimators = [hkf.HKF(x0i, tune.P0/2, tune.Q, tune.R) for x0i in x0_est_sp]
    estimator = gmf.GMF(estimators, max_filter_count=8)


    estimated_target_obj = mt_obj.Sat(x0_est[:3], area=10, reflectivity=0.9)

    est_xhist = [x0_est]
    est_Phist = [tune.P0]

    err = []
    for i in range(xhist_sat1.shape[1]-1):
        curr_state = xhist_sat1[:,i]
        next_state = xhist_sat1[:,i+1]
        target_rel_state = xhist_sat2[:,i+1] - next_state


        meas, theta = measurements[i]

        f, F = dn.rel_dyn(curr_state)
        xhat, Pk = estimator.prediction(dt, f, F)
        if meas is not None:
            for y in meas:
                h, _ = mt.gen_h_rel(next_state, theta, estimated_target_obj)
                xhat, Pk = estimator.measurement(y, h, tune.R)
        else:
            h = mt_ni.gen_h_rel(next_state, theta, estimated_target_obj)
            xhat, Pk = estimator.measurement(-np.ones(1), h, tune.R_ni, True)
        
        err.append(estimator.map_estimate()[0] - target_rel_state)
    return np.array(err)

if __name__ == "__main__":
    dt = 1
    # SS:MM:HH
    duration = 60 * 30 * 1

    results = None
    N = 32
    k = 0
    seeds = list(range(k,k+N))
    # seeds[14] = 33
    args = [(seed, dt, duration) for seed in seeds]
    err_results = process_map(run, args, max_workers=4)

    thresh = 1000
    dist_errs = np.array([np.linalg.norm(result[:,:3], axis=1) for result in err_results])
    converged = np.sum(dist_errs < thresh, axis=0) / N

    pltstyle.use(['fast'])
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=2)

    ax = axs[0]
    ax.set_yscale('log')
    tspan = np.arange(err_results[0].shape[0])
    for sim in range(N):
        ax.plot(tspan, dist_errs[sim,:])
    ax.axhline(y=thresh, color='g', linestyle='--')
    ax.set_ylabel("Position Error")
    ax.set_xlabel("Time")
    ax.set_title(f"N={N} Simulations")

    ax = axs[1]
    ax.plot(tspan, converged)
    ax.set_ylim([0,1])
    ax.set_ylabel("% Simulations Within Threshold")
    ax.set_xlabel("Time")
    ax.set_title(f"Simulations Within Error Threshold")

    plt.show()
