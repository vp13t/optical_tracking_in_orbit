import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from . import constants as c

def gravity_dyn(t, x):
    r = np.linalg.norm(x[:3])
    mu = c.G * c.earth_mass
    dvdt = -mu * x[:3] / r**3
    return np.concatenate((x[3:], dvdt))

def propagate_dyn(x, t, res=100, Q=np.zeros((6,6))):
    sol = solve_ivp(fun=gravity_dyn, t_span=(0,t), y0=x, dense_output=True, max_step=0.01)
    t_plot = np.linspace(0, t, res)
    x_plot = sol.sol(t_plot)
    noise = np.random.multivariate_normal(np.zeros(6), Q, size=1) * t
    x_plot += noise.T * np.arange(x_plot.shape[1]) / x_plot.shape[1]
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
    r = np.linalg.norm(x[:3])
    dvdx = np.zeros(3)
    mu = c.G * c.earth_mass 
    for i in range(3):
        dvdx[i] = mu * ((1 / r**3) - (3 * x[i]**2 / r**5))
        if dvdx[i] == 0:
            dvdx[i] = 1e-9
    a = mu * ((3 * x.T @ x / r**5) - np.eye(3) / r**3)
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:,:3] = a
    # F = expm(A * dt)
    F = np.eye(6) + A * dt
    return F
