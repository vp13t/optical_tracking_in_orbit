import matplotlib.pyplot as plt
import numpy as np

from measurement import rotation as rot

def plot(t_span, tracker_hist, target_hist, est_hist):
    inertial_frame = rot.Frame(np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.zeros(3))
    sat2_frames = [rot.RTNFrame(x[0:3], x[3:6]) for x in target_hist.T]
    sat2_transforms = [rot.rotation_matrix(inertial_frame, frame) for frame in sat2_frames]

    sat2_pos = np.array(target_hist)[0:3,:]

    sat1_pos_rel = np.array(tracker_hist)[0:3,:] - sat2_pos
    sat1_pos_cw = np.array([matrix @ pos for matrix, pos in zip(sat2_transforms, sat1_pos_rel.T)])
    sat1_xs = sat1_pos_cw[:,0]
    sat1_ys = sat1_pos_cw[:,1]
    sat1_zs = sat1_pos_cw[:,2]

    est_pos_rel = np.array(est_hist)[0:3,:] - sat2_pos
    est_pos_cw = np.array([matrix @ pos for matrix, pos in zip(sat2_transforms, est_pos_rel.T)])
    est_xs = est_pos_cw[:,0]
    est_ys = est_pos_cw[:,1]
    est_zs = est_pos_cw[:,2]

    sat2_xs = np.zeros_like(sat1_xs)
    sat2_ys = np.zeros_like(sat1_ys)
    sat2_zs = np.zeros_like(sat1_zs)

    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Target RTN Frame Plots')
    ax = axs[0]
    ax.scatter(sat2_xs, sat2_ys, label='Satellite 2', c='orange')
    ax.plot(sat1_xs, sat1_ys, label='Satellite 1', c='red')
    ax.scatter(sat1_xs[0], sat1_ys[0], label='Satellite 1 Start', c='red', marker='x')
    ax.plot(est_xs, est_ys, label='Estimated Position', c='green', linestyle='dashed')
    ax.scatter(est_xs[0], est_ys[0], label='Estimated Position Start', c='green', marker='x')
    ax.set_xlabel('R Position (m)')
    ax.set_ylabel('T Position (m)')
    ax.set_title('R vs T')

    ax = axs[1]
    ax.scatter(sat2_xs, sat2_zs, label='Satellite 2', c='orange')
    ax.plot(sat1_xs, sat1_zs, label='Satellite 1', c='red')
    ax.scatter(sat1_xs[0], sat1_zs[0], label='Satellite 1 Start', c='red', marker='x')
    ax.plot(est_xs, est_zs, c='green', label='Estimated Position', linestyle='dashed')
    ax.scatter(est_xs[0], est_zs[0], c='green', label='Estimated Position Start', marker='x')    # ax.scatter(est_xs, est_z
    ax.set_xlabel('R Position (m)')
    ax.set_ylabel('N Position (m)')
    ax.set_title('R vs N')

    ax = axs[2]
    ax.scatter(sat2_ys, sat2_zs, label='Satellite 2', c='orange')
    ax.plot(sat1_ys, sat1_zs, label='Satellite 1', c='red')
    ax.scatter(sat1_ys[0], sat1_zs[0], label='Satellite 1 Start', c='red', marker='x')
    ax.plot(est_ys, est_zs, c='green', label='Estimated Position', linestyle='dashed')
    ax.scatter(est_ys[0], est_zs[0], c='green', label='Estimated Position Start', marker='x')
    ax.set_xlabel('T Position (m)')
    ax.set_ylabel('N Position (m)')
    ax.set_title('T vs N')

    ax.legend()