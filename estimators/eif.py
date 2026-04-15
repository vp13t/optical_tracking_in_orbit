import numpy as np

class EIF:
    def __init__(self, x0, I0, Q, R):
        # State
        self.x = x0
        # Dimension
        self.n = x0.shape[0]
        # Information Matrix
        self.I = I0
        # Process Noise
        self.Q = Q
        self.Qinv = np.linalg.inv(Q)
        # Measurement Noise
        self.R = R
        self.Rinv = np.linalg.inv(R)

    def prediction(self, dt, f, F):
        Fk = F(self.x, dt)
        # Finv = np.linalg.inv(Fk)
        # M = Finv.T @ self.I @ Finv
        P = np.linalg.inv(self.I)
        P_pred = Fk @ P @ Fk.T + self.Q

        self.x = f(self.x, dt)
        # self.I = M - (M @ np.linalg.inv(M + self.Qinv) @ M)
        # P = np.linalg.inv(self.I)
        self.I = np.linalg.inv(P_pred)
        return self.x, P

    def measurement(self, y, h, H):
        Hk = H(self.x)

        # EIF
        innov = (y - h(self.x) + Hk @ self.x).flatten()
        i = self.I @ self.x + Hk.T @ self.Rinv @ innov
        self.I = self.I + Hk.T @ self.Rinv @ Hk
        P = np.linalg.inv(self.I)
        self.x = P @ i

        # EKF
        # P = np.linalg.inv(self.I)
        # K = P @ H.T @ np.linalg.inv(H @ P @ H.T + self.R)
        # innov = (y - h(self.x)).flatten()
        # self.x = self.x + (K @ innov).squeeze()
        # P = (np.eye(6) - K @ H) @ P
        # self.I = np.linalg.inv(P)

        return self.x, P

    def NEES(self, x):
        ex = x - self.x
        P = np.linalg.inv(self.I)
        NEES = ex.T @ P @ ex
        return NEES

    def pm2sigma(self):
        P = np.linalg.inv(self.I)
        sig = np.sqrt(np.diag(P))
        return [self.x - 2*sig, self.x + 2*sig]
