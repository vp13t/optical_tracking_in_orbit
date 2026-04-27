import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from dynamics import constants as dn_c
from measurement import stars as stars

def plot_gaussian(mu, P, ax):
    evals, evecs = np.linalg.eigh(P)
    return plot_ellipsoid(mu, evals, evecs, ax)

def plot_ellipsoid(center, radii, rotation_matrix, ax, res=50, **plot_args):
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    points = np.vstack((x.flatten() * radii[0], 
                        y.flatten() * radii[1], 
                        z.flatten() * radii[2]))    
    rotated_points = rotation_matrix @ points
    x_new, y_new, z_new = rotated_points + center[:, np.newaxis]
    x_new = x_new.reshape(x.shape)
    y_new = y_new.reshape(y.shape)
    z_new = z_new.reshape(z.shape)

    return ax.plot_wireframe(x_new, y_new, z_new, **plot_args)

def plot(tracker_hist, target_hist, est_hist, r0, debug=False, plot_stars=False):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.set_position((0, 0, 1, 1))
    if debug:
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X", labelpad=20)
        ax.set_ylabel("Y", labelpad=20)
        ax.set_zlabel("Z", labelpad=20)
        ax.set_xticks([-dn_c.earth_rad,0,dn_c.earth_rad], ["-6.37e6", "0", "6.37e6"])
        ax.set_yticks([-dn_c.earth_rad,0,dn_c.earth_rad], ["-6.37e6", "0", "6.37e6"])
        ax.set_zticks([-dn_c.earth_rad,0,dn_c.earth_rad], ["-6.37e6", "0", "6.37e6"])
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)
        ax.tick_params(axis='z', rotation=45)
        ax.set(xlim3d=(-r0, r0))
        ax.set(ylim3d=(-r0, r0))
        ax.set(zlim3d=(-r0, r0))
        ax.view_init(elev=0, azim=90)
        plot_ellipsoid(np.zeros(3), np.ones(3) * dn_c.earth_rad, np.eye(3), ax, 50)
    else:
        ax.grid(False)
        ax.axis('off')
        ax.set_facecolor('black')
        ax.scatter([0],[0],[0])

        if stars:
            star_xs = [star.pos[0] for star in stars.STARS]
            star_ys = [star.pos[1] for star in stars.STARS]
            star_zs = [star.pos[2] for star in stars.STARS]
            ax.scatter(star_xs, star_ys, star_zs, s=1, c='white')

    sat_plots = [None, None, None]
    lines = [ax.plot([], [], [])[0] for _ in range(3)]

    def update_plot(num, tracker_hist, target_hist, est_hist, lines, sat_plots, ax):
        for sat_plot in sat_plots:
            if sat_plot is not None:
                sat_plot.remove()
                sat_plot = None

        tracker_x1k = tracker_hist[:3, :num+1]
        target_x1k = target_hist[:3, :num+1]
        est_x1k = est_hist[:3, :num+1]
        if lines:
            lines[0].set_data_3d(tracker_x1k)
            lines[1].set_data_3d(target_x1k)
            lines[2].set_data_3d(est_x1k)
        tracker_xk = tracker_x1k[:,-1]
        target_xk = target_x1k[:,-1]
        est_xk = est_x1k[:,-1]
        sat_plots[0] = ax.scatter(tracker_xk[0], tracker_xk[1], tracker_xk[2], c="blue")
        sat_plots[1] = ax.scatter(target_xk[0], target_xk[1], target_xk[2], c="orange")
        sat_plots[2] = ax.scatter(est_xk[0], est_xk[1], est_xk[2], c="green")

        if not debug:
            ax.set(xlim3d=(target_xk[0]-10000, target_xk[0]+10000), xlabel='X')
            ax.set(ylim3d=(target_xk[1]-10000, target_xk[1]+10000), ylabel='Y')
            ax.set(zlim3d=(target_xk[2]-10000, target_xk[2]+10000), zlabel='Z')

        return lines

    ani = animation.FuncAnimation(
        fig, update_plot, target_hist.shape[1], fargs=(tracker_hist, target_hist, est_hist, lines, sat_plots, ax), interval=10, repeat=False
    )

    plt.show()
