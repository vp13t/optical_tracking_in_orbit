import numpy as np
from scipy.optimize import approx_fprime

from . import rotation as rot
from . import constants as c
from . import objects as obj

def camera_frame(cam_state, theta):
    cam_pos = cam_state[:3]
    cam_vel = cam_state[3:]

    observer_frame = rot.RTNFrame(cam_pos, cam_vel)
    theta = theta % (2*np.pi)
    cam_Y = rot.rotate_vector(observer_frame.y_axis, observer_frame.x_axis, theta)
    cam_Z = rot.rotate_vector(observer_frame.z_axis, observer_frame.x_axis, theta)
    cam_frame = rot.Frame(observer_frame.x_axis, cam_Y, cam_Z, cam_pos)
    return cam_frame

def gen_camera_image(cam_state, theta, obs_pts):
    """
    Theta is camera rotation counterclockwise about the radial axis.
    """
    cam_frame = camera_frame(cam_state, theta)
    cam_pos = cam_frame.origin
    Phi = rot.rotation_matrix(rot.InertialFrame(), cam_frame)

    view = np.zeros(c.view_px)
    view_center = cam_frame.y_axis

    for obs_pt in obs_pts:
        obs_vec = Phi @ (obs_pt.pos - cam_pos)
        obs_dist = np.linalg.norm(obs_vec)
        # obs_dir = obs_vec / obs_dist

        # x_theta = rot.signed_angle_vector_plane(obs_dir, cam_frame.x_axis)
        # z_theta = rot.signed_angle_vector_plane(obs_dir, cam_frame.z_axis)
        x_theta = np.atan2(obs_vec[0], obs_vec[1])
        z_theta = np.atan2(obs_vec[2], obs_vec[1])
        # dot = np.dot(obs_dir, view_center)

        if np.abs(z_theta) > c.view_angle/2 or np.abs(x_theta) > c.view_angle/2:
        # if np.abs(z_theta) > c.view_angle/2 or np.abs(x_theta) > c.view_angle/2 or dot < 0:
            continue

        xidx = int(np.floor(x_theta * c.view_px[0] / c.view_angle) + c.view_px[0]/2)
        zidx = int(np.floor(z_theta * c.view_px[1] / c.view_angle) + c.view_px[1]/2)

        view[xidx, zidx] = int(obs_pt.brightness(obs_dist, c.view_angle/c.view_px[0]))
    return view

def meas_from_camera_image(image):
    bright_coords = np.argwhere(image > 0)
    if bright_coords.shape[0] == 0:
        return None
    brightnesses = image[bright_coords[:,0], bright_coords[:,1]]
    observations = [np.array([bright_coords[i][0], bright_coords[i][1], np.log(brightnesses[i])]) for i in range(bright_coords.shape[0])]
    return observations

def gen_h(cam_state, theta, sat):
    cam_frame = camera_frame(cam_state, theta)
    Phi = rot.rotation_matrix(rot.InertialFrame(), cam_frame)

    px_multi = c.view_px[0] / c.view_angle
    pz_multi = c.view_px[1] / c.view_angle
    brightness_multi = sat.reflectivity * sat.area * 255 / (2*np.tan(c.view_angle/c.view_px[0]))**2

    def h(x):
        xc = Phi @ (x[:3] - cam_state[:3])
        Xc = xc[0]
        Yc = xc[1]
        Zc = xc[2]

        y = np.zeros(3)
        y[0] = np.atan2(Xc, Yc) * px_multi + c.view_px[0]/2
        y[1] = np.atan2(Zc, Yc) * pz_multi + c.view_px[1]/2
        y[2] = np.log(brightness_multi / (Xc**2 + Yc**2 + Zc**2))
        return y
    return h

def gen_h_rel(cam_state, theta, sat):
    cam_frame = camera_frame(cam_state, theta)
    Phi = rot.rotation_matrix(rot.InertialFrame(), cam_frame)

    px_multi = c.view_px[0] / c.view_angle
    pz_multi = c.view_px[1] / c.view_angle
    brightness_multi = sat.reflectivity * sat.area * 255 / (2*np.tan(c.view_angle/c.view_px[0]))**2

    def h(x):
        xc = Phi @ x[:3]
        Xc = xc[0]
        Yc = xc[1]
        Zc = xc[2]

        y = np.zeros(3)
        y[0] = np.atan2(Xc, Yc) * px_multi + c.view_px[0]/2
        y[1] = np.atan2(Zc, Yc) * pz_multi + c.view_px[1]/2
        y[2] = np.log(brightness_multi / (Xc**2 + Yc**2 + Zc**2))
        return y
    return h

def gen_H_rel(cam_state, theta, sat):
    cam_frame = camera_frame(cam_state, theta)
    Phi = rot.rotation_matrix(rot.InertialFrame(), cam_frame)

    px_multi = c.view_px[0] / c.view_angle
    pz_multi = c.view_px[1] / c.view_angle
    brightness_multi = -2 * sat.reflectivity * sat.area * 255 / (2*np.tan(c.view_angle/c.view_px[0]))**2

    def H(x):
        # xc = Phi @ (x[:3] - cam_state[:3])
        xc = Phi @ x[:3]
        Xc = xc[0]
        Yc = xc[1]
        Zc = xc[2]

        Hk = np.zeros((3,6))
        Hk[0, 0] = px_multi * (Yc*Phi[0,0] - Xc*Phi[1,0]) / (Xc**2 + Yc**2)
        Hk[0, 1] = px_multi * (Yc*Phi[0,1] - Xc*Phi[1,1]) / (Xc**2 + Yc**2)
        Hk[0, 2] = px_multi * (Yc*Phi[0,2] - Xc*Phi[1,2]) / (Xc**2 + Yc**2)
        Hk[1, 0] = pz_multi * (Yc*Phi[2,0] - Zc*Phi[1,0]) / (Zc**2 + Yc**2)
        Hk[1, 1] = pz_multi * (Yc*Phi[2,1] - Zc*Phi[1,1]) / (Zc**2 + Yc**2)
        Hk[1, 2] = pz_multi * (Yc*Phi[2,2] - Zc*Phi[1,2]) / (Zc**2 + Yc**2)
        Hk[2, 0] = -2*(Xc*Phi[0,0] + Yc*Phi[0,1] + Zc*Phi[0,2]) / (Xc**2 + Yc**2 + Zc**2)
        Hk[2, 1] = -2*(Xc*Phi[1,0] + Yc*Phi[1,1] + Zc*Phi[1,2]) / (Xc**2 + Yc**2 + Zc**2)
        Hk[2, 2] = -2*(Xc*Phi[2,0] + Yc*Phi[2,1] + Zc*Phi[2,2]) / (Xc**2 + Yc**2 + Zc**2)
        print(Hk)
        return Hk
    return H

if __name__ == "__main__":
    r0 = 6371000 + 300000

    sat2_z0 = r0 + 500
    sat2_y0 = 1000
    sat2_x0 = 7000
    sat2_r0_approx = np.array([sat2_x0, sat2_y0, sat2_z0])
    sat2_r0_n = sat2_r0_approx / np.linalg.norm(sat2_r0_approx)
    sat2_r0 = sat2_z0 * sat2_r0_n

    x0_sat1 = np.array([-500, -1000, r0, 7500, 0, 0])
    x0_sat2 = np.concatenate((sat2_r0, [7500, 0, 0]))
    cam_state = x0_sat1
    theta = np.deg2rad(15)

    target_obj = obj.Sat(x0_sat2[:3], area=10, reflectivity=0.9)

    im = gen_camera_image(cam_state, theta, [target_obj])
    meas = meas_from_camera_image(im)
    print(meas)
    h = gen_h(cam_state, theta, target_obj)
    print(h(x0_sat2))

    def jacobian(h, x, eps=1e-5):
        return approx_fprime(x, h, eps)

    H = gen_H(cam_state, theta, target_obj)
    print(H(x0_sat2))
    J = jacobian(h, x0_sat2)
    print(J)
    print(H(x0_sat2)-J)