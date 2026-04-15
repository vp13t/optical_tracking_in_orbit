import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from . import constants as c

def gravity_dyn(t, x):
    r = np.linalg.norm(x[:3])
    mu = c.G * c.earth_mass
    dvdt = -mu * x[:3] / r**3
    return np.concatenate((x[3:], dvdt))

def propagate_dyn(x0, dt, duration, Q=np.zeros((6,6))):
    timesteps = int(duration/dt)
    t_plot = [0]
    x_plot = [x0]

    curr_x = x0
    for tk in range(timesteps):
        sol = solve_ivp(fun=gravity_dyn, t_span=(0,dt), y0=curr_x, dense_output=True, max_step=0.01)
        t_plot.append((tk+1) * dt)
        x = sol.y[:,-1].flatten()
        x_plot.append(x)
        noise = np.random.multivariate_normal(np.zeros(6), Q, size=1).flatten()
        curr_x = x + noise
    x_plot = np.array(x_plot).T
    t_plot = np.array(t_plot)
    return t_plot, x_plot

def rel_dyn(x_sat1):
    def f_rel(dx, dt):
        x_sat2 = dx + x_sat1
        # print(f"  f_rel: |dx|={np.linalg.norm(dx[:3]):.1f}, |x_sat2|={np.linalg.norm(x_sat2[:3]):.1f}, |x_sat1|={np.linalg.norm(x_sat1[:3]):.1f}")
        result = f(x_sat2, dt) - f(x_sat1, dt)
        # print(f"  f_rel result: {result}")
        return result
        # return f(x_sat2, dt) - f(x_sat1, dt)
    def F_rel(dx, dt):
        x_sat2 = dx + x_sat1
        return F(x_sat2, dt)
    return f_rel, F_rel

def f(x, dt):
    f_x = solve_ivp(fun=gravity_dyn, t_span=(0,dt), y0=x, dense_output=True, max_step=0.01)
    return f_x.y[:,-1]

def F(x, dt):
    r_vec = x[:3]
    r = np.linalg.norm(r_vec)
    dvdx = np.zeros(3)
    mu = c.G * c.earth_mass 
    a = mu * ((3 * np.outer(r_vec, r_vec) / r**5) - np.eye(3) / r**3)
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:,:3] = a
    F = expm(A * dt)
    # F = np.eye(6) + A * dt
    return F
