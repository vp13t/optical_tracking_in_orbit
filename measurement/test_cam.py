"""
Test of camera rotation.

A point cloud is used as a reference visible to the naked eye at 1024x1024 resolution.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from . import measurement as ms
from . import objects as obj
from . import rotation as rot

if __name__ == "__main__":
    cam = np.array([0,0,2,1,0,0])
    mu1 = [2,0,2]
    sigma = np.array([[1,0,0], [0,1,-0.5], [0,-0.5,1]]) * 0.005
    samples1 = np.random.multivariate_normal(mu1, sigma, size=1000)

    mu2 = [3,1,3]
    samples2 = np.random.multivariate_normal(mu2, sigma, size=1000)

    samples = np.concatenate((samples1, samples2), axis=0)
    pts = [obj.ObsPt(sample) for sample in samples]


    snapshots = []
    for k in range(0,405,5):
        meas = ms.gen_camera_image(cam,np.deg2rad(k),pts)
        snapshots.append(meas.T)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(samples[:,0], samples[:,1], samples[:,2])
    ax.scatter(cam[0], cam[1], cam[2], color='red', label='Camera')
    observer_frame = rot.RTNFrame(cam[0:3], cam[3:6])
    ax.plot([cam[0], cam[0] + observer_frame.x_axis[0]], [cam[1], cam[1] + observer_frame.x_axis[1]], [cam[2], cam[2] + observer_frame.x_axis[2]], color='red', label='Camera X-axis')
    ax.plot([cam[0], cam[0] + observer_frame.y_axis[0]], [cam[1], cam[1] + observer_frame.y_axis[1]], [cam[2], cam[2] + observer_frame.y_axis[2]], color='green', label='Camera Y-axis')
    ax.plot([cam[0], cam[0] + observer_frame.z_axis[0]], [cam[1], cam[1] + observer_frame.z_axis[1]], [cam[2], cam[2] + observer_frame.z_axis[2]], color='blue', label='Camera Z-axis')

    theta = np.deg2rad(15)
    cam_Y = rot.rotate_vector(observer_frame.y_axis, observer_frame.x_axis, theta)
    cam_Z = rot.rotate_vector(observer_frame.z_axis, observer_frame.x_axis, theta)
    cam_frame = rot.Frame(observer_frame.x_axis, cam_Y, cam_Z, cam[0:3])
    ax.plot([cam[0], cam[0] + cam_frame.y_axis[0]], [cam[1], cam[1] + cam_frame.y_axis[1]], [cam[2], cam[2] + cam_frame.y_axis[2]], color='orange', label='Rotated Camera Y-axis')
    ax.plot([cam[0], cam[0] + cam_frame.z_axis[0]], [cam[1], cam[1] + cam_frame.z_axis[1]], [cam[2], cam[2] + cam_frame.z_axis[2]], color='purple', label='Rotated Camera Z-axis')

    ax.legend()

    fig, ax = plt.subplots()
    im = ax.imshow(snapshots[0], animated=True, cmap='viridis', vmin=0, vmax=255, origin='lower')
    ax.axis('off')

    animate = False
    if animate:
        def animate_func(i):
            """Updates the image data for frame i."""
            ax.set_title(f'Frame Number: {i*5}')
            im.set_array(snapshots[i])
            return [im]

        # 4. Create the animation
        ani = animation.FuncAnimation(
            fig, 
            animate_func, 
            frames=len(snapshots),
            interval=50,
            blit=False,
        )    
    plt.show()