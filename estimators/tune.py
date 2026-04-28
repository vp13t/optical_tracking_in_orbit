import numpy as np

# More accurate tracking (60%), but inconsistent due to outliers
Q = 2.0 * np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
R = 2.0 * np.diag([1.0, 1.0, 1.0])
R_ni = 1.0 * np.diag([1.0])
P0 = 1.0 * (np.diag([5000.0, 5000.0, 5000.0, 10.0, 10.0, 10.0]) ** 2)

# Fairly consistent, but inaccurate at tracking
# Q = 50.0 * (np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]) ** 2)
# R = 5.0 * (np.diag([100.0, 100.0, 10.0]) ** 2)
# R_ni = 10.0 * (np.diag([2.0]) ** 2)
# P0 = 1.0 * (np.diag([5000.0, 5000.0, 5000.0, 100.0, 100.0, 100.0]) ** 2)

PG = 0.8
PD = 0.9
Lambda = 10.0