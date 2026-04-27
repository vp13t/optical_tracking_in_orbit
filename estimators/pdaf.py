
import numpy as np
from scipy.stats import chi2
from enum import Enum
from scipy.stats import multivariate_normal

from . import hkf

class PDAF(hkf.HKF):
    """Probabilistic Data Association Filter"""
    def __init__(self, x0, P0, Q, R, PG, PD, Lambda):
        super().__init__(x0, P0, Q, R)

        # Gate Probability
        self.PG = PG
        # Detection Probability
        self.PD = PD
        # False Detection Poisson Density
        self.Lambda = Lambda

    def measurement(self, y, h, R):
        if y is None:
            return self.x, self.P
        return super().measurement(y, h, R)

    def pda(self, ys, h, R):
        yhat, innov_cov = self.innov(h, R)

        validated_idxs = []
        validated_likelihoods = []
        for j in range(len(ys)):
            nis = (ys[j] - yhat).T @ np.linalg.inv(innov_cov) @ (ys[j] - yhat)
            # Validation Threshold
            gamma = chi2.ppf(q=self.PG, df=ys[j].shape[0])
            if nis <= gamma:
                validated_idxs.append(j)
                likelihood = multivariate_normal.pdf(ys[j], mean=yhat, cov=innov_cov) * self.PD / self.Lambda
                validated_likelihoods.append(likelihood)
        ys = [ys[j] for j in validated_idxs]
        return ys, validated_likelihoods

    def pda_ni(self, ys, h, R, y_null, h_null, R_null):
        ys, likelihoods = self.pda(ys, h, R)
        hs = [h] * len(ys)
        Rs = [R] * len(ys)

        if not ys:
            ys, likelihoods = self._pda([y_null], h_null, R_null)
            hs = [h_null] * len(ys)
            Rs = [R_null] * len(ys)

        missed_detection_likelihood = 1 - self.PG*self.PD
        ys.append(None)
        hs.append(None)
        Rs.append(None)
        likelihoods.append(missed_detection_likelihood)

        likelihood_sum = np.sum(likelihoods)
        likelihoods = [p/likelihood_sum for p in likelihoods]
        return ys, hs, Rs, likelihoods
