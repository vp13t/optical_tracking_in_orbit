import matplotlib.pyplot as plt
import numpy as np
from measurement import rotation as rot
from scipy.stats import norm

def plot(sat1_xhist, sat2_xhist, est_xhists, est_Phists, weights_hist=None):
    est_xhists = [np.array([est_xhists[k][j] for k in range(len(est_xhists))]).T for j in range(len(est_xhists[0]))]
    est_Phists = [[est_Phists[k][j] for k in range(len(est_Phists))] for j in range(len(est_Phists[0]))]

    inertial_frame = rot.InertialFrame()
    sat1_frames = [rot.RTNFrame(x[0:3], x[3:6]) for x in sat1_xhist.T]
    sat1_transforms = [rot.rotation_matrix(inertial_frame, frame) for frame in sat1_frames]

    errs_pos_sat1frame = []
    errs_vel_sat1frame = []
    covs_all_pos_sat1frame = []
    covs_all_vel_sat1frame = []
    for est_xhist, est_Phist in zip(est_xhists, est_Phists):
        err_hist = est_xhist - sat2_xhist
        err_pos_sat1frame = np.array([matrix @ err for matrix, err in zip(sat1_transforms, err_hist[:3,:].T)])
        err_vel_sat1frame = np.array([matrix @ err for matrix, err in zip(sat1_transforms, err_hist[3:,:].T)])

        est_pos_Phist = [cov[:3,:3] for cov in est_Phist]
        est_vel_Phist = [cov[3:,3:] for cov in est_Phist]
        covs_pos_sat1frame = [matrix @ cov @ matrix.T for matrix, cov in zip(sat1_transforms, est_pos_Phist)]
        covs_vel_sat1frame = [matrix @ cov @ matrix.T for matrix, cov in zip(sat1_transforms, est_vel_Phist)]

        errs_pos_sat1frame.append(err_pos_sat1frame)
        errs_vel_sat1frame.append(err_vel_sat1frame)
        covs_all_pos_sat1frame.append(covs_pos_sat1frame)
        covs_all_vel_sat1frame.append(covs_vel_sat1frame)

    if len(est_xhists) == 1:
        pos_ranges = np.sqrt(np.diag(covs_all_pos_sat1frame[0][0])) * 2
        vel_ranges = np.sqrt(np.diag(covs_all_vel_sat1frame[0][0])) * 2
    else:
        pos_ranges = np.sqrt(np.diag(covs_all_pos_sat1frame[0][0])) * 4
        vel_ranges = np.sqrt(np.diag(covs_all_vel_sat1frame[0][0])) * 4

    xspan = np.linspace(-pos_ranges[0],pos_ranges[0],100)
    yspan = np.linspace(-pos_ranges[1],pos_ranges[1],100)
    zspan = np.linspace(-pos_ranges[2],pos_ranges[2],100)
    xdotspan = np.linspace(-vel_ranges[0],vel_ranges[0],100)
    ydotspan = np.linspace(-vel_ranges[1],vel_ranges[1],100)
    zdotspan = np.linspace(-vel_ranges[2],vel_ranges[2],100)

    tmax = est_xhists[0].shape[1]
    tspan = range(tmax)

    px_all_weighted = np.zeros((100, tmax))
    py_all_weighted = np.zeros((100, tmax))
    pz_all_weighted = np.zeros((100, tmax))
    pxdot_all_weighted = np.zeros((100, tmax))
    pydot_all_weighted = np.zeros((100, tmax))
    pzdot_all_weighted = np.zeros((100, tmax))

    px_all_unweighted = np.zeros((100, tmax))
    py_all_unweighted = np.zeros((100, tmax))
    pz_all_unweighted = np.zeros((100, tmax))
    pxdot_all_unweighted = np.zeros((100, tmax))
    pydot_all_unweighted = np.zeros((100, tmax))
    pzdot_all_unweighted = np.zeros((100, tmax))

    for i in range(len(est_xhists)):
        px, py, pz, pxdot, pydot, pzdot = [], [], [], [], [], []

        err_pos_sat1frame = errs_pos_sat1frame[i]
        err_vel_sat1frame = errs_vel_sat1frame[i]
        cov_pos_sat1frame = covs_all_pos_sat1frame[i]
        cov_vel_sat1frame = covs_all_vel_sat1frame[i]
        for k in tspan:
            pxk = norm.pdf(xspan, loc=err_pos_sat1frame[k,0], scale=np.sqrt(cov_pos_sat1frame[k][0,0]))
            pyk = norm.pdf(yspan, loc=err_pos_sat1frame[k,1], scale=np.sqrt(cov_pos_sat1frame[k][1,1]))
            pzk = norm.pdf(zspan, loc=err_pos_sat1frame[k,2], scale=np.sqrt(cov_pos_sat1frame[k][2,2]))
            pxdotk = norm.pdf(xdotspan, loc=err_vel_sat1frame[k,0], scale=np.sqrt(cov_vel_sat1frame[k][0,0]))
            pydotk = norm.pdf(ydotspan, loc=err_vel_sat1frame[k,1], scale=np.sqrt(cov_vel_sat1frame[k][1,1]))
            pzdotk = norm.pdf(zdotspan, loc=err_vel_sat1frame[k,2], scale=np.sqrt(cov_vel_sat1frame[k][2,2]))
            px.append(pxk/max(np.max(pxk),1e-100))
            py.append(pyk/max(np.max(pyk),1e-100))
            pz.append(pzk/max(np.max(pzk),1e-100))
            pxdot.append(pxdotk/max(np.max(pxdotk),1e-100))
            pydot.append(pydotk/max(np.max(pydotk),1e-100))
            pzdot.append(pzdotk/max(np.max(pzdotk),1e-100))
        weights = weights_hist[:,i] if weights_hist is not None else np.ones((1,tmax))
        px_all_weighted += np.column_stack(px) * weights
        py_all_weighted += np.column_stack(py) * weights
        pz_all_weighted += np.column_stack(pz) * weights
        pxdot_all_weighted += np.column_stack(pxdot) * weights
        pydot_all_weighted += np.column_stack(pydot) * weights
        pzdot_all_weighted += np.column_stack(pzdot) * weights
        px_all_unweighted += np.column_stack(px)
        py_all_unweighted += np.column_stack(py)
        pz_all_unweighted += np.column_stack(pz)
        pxdot_all_unweighted += np.column_stack(pxdot)
        pydot_all_unweighted += np.column_stack(pydot)
        pzdot_all_unweighted += np.column_stack(pzdot)
    
    px_all_unweighted_log = np.log(px_all_unweighted + 1)
    py_all_unweighted_log = np.log(py_all_unweighted + 1)
    pz_all_unweighted_log = np.log(pz_all_unweighted + 1)
    pxdot_all_unweighted_log = np.log(pxdot_all_unweighted + 1)
    pydot_all_unweighted_log = np.log(pydot_all_unweighted + 1)
    pzdot_all_unweighted_log = np.log(pzdot_all_unweighted + 1)

    fig, axs = plt.subplots(3, figsize=(6, 6))
    plt.tight_layout(pad=4.0)
    im = axs[0].imshow(pxdot_all_weighted, extent=[tspan[0],tspan[-1],-vel_ranges[0],vel_ranges[0]], aspect='auto', cmap='magma')
    axs[0].set_title('$\\dot{R}_{tracker}$ Error Estimate Probability Mass')
    axs[0].set_xlabel("Time")
    im = axs[1].imshow(pydot_all_weighted, extent=[tspan[0],tspan[-1],-vel_ranges[1],vel_ranges[1]], aspect='auto', cmap='magma')
    axs[1].set_title('$\\dot{T}_{tracker}$ Error Estimate Probability Mass')
    axs[1].set_xlabel("Time")
    im = axs[2].imshow(pzdot_all_weighted, extent=[tspan[0],tspan[-1],-vel_ranges[2],vel_ranges[2]], aspect='auto', cmap='magma')
    axs[2].set_title('$\\dot{N}_{tracker}$ Error Estimate Probability Mass')
    axs[2].set_xlabel("Time")

    fig, axs = plt.subplots(3, figsize=(6, 6))
    plt.tight_layout(pad=4.0)
    im = axs[0].imshow(px_all_weighted, extent=[tspan[0],tspan[-1],-pos_ranges[0],pos_ranges[0]], aspect='auto', cmap='magma')
    axs[0].set_title('${R}_{tracker}$ Error Estimate Probability Mass')
    axs[0].set_xlabel("Time")
    im = axs[1].imshow(py_all_weighted, extent=[tspan[0],tspan[-1],-pos_ranges[1],pos_ranges[1]], aspect='auto', cmap='magma')
    axs[1].set_title('${T}_{tracker}$ Error Estimate Probability Mass')
    axs[1].set_xlabel("Time")
    im = axs[2].imshow(pz_all_weighted, extent=[tspan[0],tspan[-1],-pos_ranges[2],pos_ranges[2]], aspect='auto', cmap='magma')
    axs[2].set_title('${N}_{tracker}$ Error Estimate Probability Mass')
    axs[2].set_xlabel("Time")

    fig, axs = plt.subplots(3, figsize=(6, 6))
    plt.tight_layout(pad=4.0)
    im = axs[0].imshow(pxdot_all_unweighted_log, extent=[tspan[0],tspan[-1],-vel_ranges[0],vel_ranges[0]], aspect='auto', cmap='magma')
    axs[0].set_title('$\\dot{R}_{tracker}$ Error Estimate Unweighted Log Probability')
    axs[0].set_xlabel("Time")
    im = axs[1].imshow(pydot_all_unweighted_log, extent=[tspan[0],tspan[-1],-vel_ranges[1],vel_ranges[1]], aspect='auto', cmap='magma')
    axs[1].set_title('$\\dot{T}_{tracker}$ Error Estimate Unweighted Log Probability')
    axs[1].set_xlabel("Time")
    im = axs[2].imshow(pzdot_all_unweighted_log, extent=[tspan[0],tspan[-1],-vel_ranges[2],vel_ranges[2]], aspect='auto', cmap='magma')
    axs[2].set_title('$\\dot{N}_{tracker}$ Error Estimate Unweighted Log Probability')
    axs[2].set_xlabel("Time")

    fig, axs = plt.subplots(3, figsize=(6, 6))
    plt.tight_layout(pad=4.0)
    im = axs[0].imshow(px_all_unweighted_log, extent=[tspan[0],tspan[-1],-pos_ranges[0],pos_ranges[0]], aspect='auto', cmap='magma')
    axs[0].set_title('${R}_{tracker}$ Error Estimate Unweighted Log Probability')
    axs[0].set_xlabel("Time")
    im = axs[1].imshow(py_all_unweighted_log, extent=[tspan[0],tspan[-1],-pos_ranges[1],pos_ranges[1]], aspect='auto', cmap='magma')
    axs[1].set_title('${T}_{tracker}$ Error Estimate Unweighted Log Probability')
    axs[1].set_xlabel("Time")
    im = axs[2].imshow(pz_all_unweighted_log, extent=[tspan[0],tspan[-1],-pos_ranges[2],pos_ranges[2]], aspect='auto', cmap='magma')
    axs[2].set_title('${N}_{tracker}$ Error Estimate Unweighted Log Probability')
    axs[2].set_xlabel("Time")
