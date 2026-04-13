from scipy.spatial.transform import Rotation
import math
import numpy as np

def angle_between(v1, v2, axis_normal):
    v1_n = v1 / np.linalg.norm(v1)
    v2_n = v2 / np.linalg.norm(v2)
    axis_normal_n = axis_normal / np.linalg.norm(axis_normal)

    s = np.linalg.norm(np.cross(v1_n, v2_n))
    c = np.dot(v1_n, v2_n)
    angle = np.arctan2(s, c)
    if np.dot(np.cross(v1_n, v2_n), axis_normal_n) < 0:
        angle = -angle
    return angle

def signed_angle_vector_plane(vector, plane_normal):
    v = vector / np.linalg.norm(vector)
    n = plane_normal / np.linalg.norm(plane_normal)

    dot_prod_v_n = np.dot(v, n)
    signed_angle = math.asin(np.clip(dot_prod_v_n, -1.0, 1.0))
    return signed_angle

def rotate_vector(vector, about_axis, angle_radians):
    axis = about_axis / np.linalg.norm(about_axis)
    rotation_vector = angle_radians * axis
    rot = Rotation.from_rotvec(rotation_vector)
    rotated_vec = rot.apply(vector)
    return rotated_vec

class Frame:
    def __init__(self, x_axis, y_axis, z_axis, origin=np.zeros(3)):
        self.x_axis = x_axis / np.linalg.norm(x_axis)
        self.y_axis = y_axis / np.linalg.norm(y_axis)
        self.z_axis = z_axis / np.linalg.norm(z_axis)
        self.origin = origin

class InertialFrame(Frame):
    def __init__(self):
        super().__init__(np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]))

class RTNFrame(Frame):
    def __init__(self, position, velocity):
        r_axis = position / np.linalg.norm(position)
        v_axis = velocity / np.linalg.norm(velocity)
        n_axis = np.cross(r_axis, v_axis)
        t_axis = np.cross(n_axis, r_axis)
        super().__init__(r_axis, t_axis, n_axis)

def rotation_matrix(original_frame, target_frame):
    R = np.zeros((3,3))
    R[0,0] = np.dot(target_frame.x_axis, original_frame.x_axis)
    R[0,1] = np.dot(target_frame.x_axis, original_frame.y_axis)
    R[0,2] = np.dot(target_frame.x_axis, original_frame.z_axis)
    R[1,0] = np.dot(target_frame.y_axis, original_frame.x_axis)
    R[1,1] = np.dot(target_frame.y_axis, original_frame.y_axis)
    R[1,2] = np.dot(target_frame.y_axis, original_frame.z_axis)
    R[2,0] = np.dot(target_frame.z_axis, original_frame.x_axis)
    R[2,1] = np.dot(target_frame.z_axis, original_frame.y_axis)
    R[2,2] = np.dot(target_frame.z_axis, original_frame.z_axis)
    return R

if __name__ == "__main__":
    I = InertialFrame()
    xhat = rotate_vector(I.x_axis, I.z_axis, np.deg2rad(45))
    yhat = rotate_vector(I.y_axis, I.z_axis, np.deg2rad(45))
    B = Frame(xhat, yhat, I.z_axis)
    Phi = rotation_matrix(InertialFrame(), B)
    print(Phi)
    x = np.ones(3)
    print(Phi @ x)
    print(Phi.T @ x)
