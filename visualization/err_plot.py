import matplotlib.pyplot as plt
import numpy as np

def plot(t_span, sat2_xhist, est_xhist, est_Phist):
    err = est_xhist - sat2_xhist

    covs = np.array([np.diag(P) for P in est_Phist]).T
    sigmas = np.sqrt(covs)

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Error Plots (Inertial Frame)')

    ax = axs[0,0]
    ax.set_yscale('log')
    ax.plot(t_span, np.abs(err[0]), label='Error', c='blue')
    ax.plot(t_span, sigmas[0] * 2, label='2 Sigma', c='red', linestyle='dashed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position (m)')
    ax.set_title('X vs Time')

    ax = axs[0,1]
    ax.set_yscale('log')
    ax.plot(t_span, np.abs(err[1]), label='Error', c='blue')
    ax.plot(t_span, sigmas[1] * 2, label='2 Sigma', c='red', linestyle='dashed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Y vs Time')

    ax = axs[0,2]
    ax.set_yscale('log')
    ax.plot(t_span, np.abs(err[2]), label='Error', c='blue')
    ax.plot(t_span, sigmas[2] * 2, label='2 Sigma', c='red', linestyle='dashed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z Position (m)')
    ax.set_title('Z vs Time')

    ax = axs[1,0]
    ax.set_yscale('log')
    ax.plot(t_span, np.abs(err[3]), label='Error', c='blue')
    ax.plot(t_span, sigmas[3] * 2, label='2 Sigma', c='red', linestyle='dashed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Velocity (m/s)')
    ax.set_title('X Velocity vs Time')

    ax = axs[1,1]
    ax.set_yscale('log')
    ax.plot(t_span, np.abs(err[4]), label='Error', c='blue')
    ax.plot(t_span, sigmas[4] * 2, label='2 Sigma', c='red', linestyle='dashed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Velocity (m/s)')
    ax.set_title('Y Velocity vs Time')

    ax = axs[1,2]
    ax.set_yscale('log')
    ax.plot(t_span, np.abs(err[5]), label='Error', c='blue')
    ax.plot(t_span, sigmas[5] * 2, label='2 Sigma', c='red', linestyle='dashed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z Velocity (m/s)')
    ax.set_title('Z Velocity vs Time')
