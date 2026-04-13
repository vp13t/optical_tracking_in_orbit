import matplotlib.pyplot as plt
import numpy as np

import dynamics as dn
import constants as c

if __name__ == "__main__":
    daySec = 60*60*24
    r0 = c.earth_rad + 300000
    v_circ = np.sqrt(c.G * c.earth_mass / r0)
    y0 = np.array([0, 0, r0, v_circ, 0, 0])

    t_plot, xplot = dn.propagate_dyn(y0, daySec, 100000)

    fig, axs = plt.subplots(1, 4)
    fig.suptitle("Position vs Time")
    axs[0].plot(t_plot, xplot[0])
    axs[0].set_title("x vs t")
    axs[1].plot(t_plot, xplot[1])
    axs[1].set_title("y vs t")
    axs[2].plot(t_plot, xplot[2])
    axs[2].set_title("z vs t")
    axs[3].plot(t_plot, np.linalg.norm(xplot[0:3,:], axis=0))
    axs[3].set_title("|r| vs t")
    plt.show()

    fig, axs = plt.subplots(1, 4)
    fig.suptitle
    axs[0].plot(t_plot, xplot[3])
    axs[0].set_title("vx vs t")
    axs[1].plot(t_plot, xplot[4])
    axs[1].set_title("vy vs t")
    axs[2].plot(t_plot, xplot[5])
    axs[2].set_title("vz vs t")
    axs[3].plot(t_plot, np.linalg.norm(xplot[3:6,:], axis=0))
    axs[3].set_title("|v| vs t")
    plt.show()

    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Phase Plots")
    axs[0].plot(xplot[0], xplot[1])
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("x vs y")
    axs[1].plot(xplot[1], xplot[2])
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("z")
    axs[1].set_title("y vs z")
    axs[2].plot(xplot[0], xplot[2])
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("z")
    axs[2].set_title("x vs z")
    plt.show()
    exit(0)