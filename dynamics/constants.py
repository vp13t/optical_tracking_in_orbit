import numpy as np

earth_rad = 6371000
earth_mass = 5.97 * 10**24
G = 6.674 * 10**-11
true_Q = np.diag([1, 1, 1, 0.01, 0.01, 0.01])