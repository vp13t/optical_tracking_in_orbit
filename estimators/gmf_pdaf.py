import numpy as np
from scipy.stats import multivariate_normal
from loky import get_reusable_executor
import copy


class GMF:
    """Gaussian Mixture Filter"""

    def __init__(self, initial_filters, max_filter_count, max_workers=4):
        self.filters = initial_filters
        self.weights = np.ones(len(initial_filters)) / len(initial_filters)
        self.max_filter_count = max_filter_count
        self.executor = get_reusable_executor(max_workers=max_workers)

        while len(self.filters) > self.max_filter_count:
            i, j = self.runnalls_kld_argmin()
            self.merge_filters(i, j)

    def prediction(self, dt, f, F):
        def pred_in_place(Fi):
            Fi.prediction(dt, f, F)
            return Fi
        self.filters = list(self.executor.map(pred_in_place, self.filters))
        return [filter.x for filter in self.filters], [filter.P for filter in self.filters]

    def measurement(self, ys, h, R, h_null, R_null):
        def measurement_hypothesis_split(Fi):
            filtered_ys, hs, Rs, likelihoods = Fi.pda_ni(ys, h, R, np.array([-1.0]), h_null, R_null)
            hypotheses = []
            for i in range(len(filtered_ys)):
                hypothesis = copy.deepcopy(Fi)
                hypothesis.measurement(filtered_ys[i], hs[i], Rs[i])
                
                hypotheses.append(hypothesis)
            return hypotheses, likelihoods
        split_hypotheses = list(self.executor.map(measurement_hypothesis_split, self.filters))
        filters = []
        weights = np.array([])
        for i in range(len(split_hypotheses)):
            hypotheses, hweights = split_hypotheses[i]
            filters.extend(hypotheses)
            weights = np.hstack((weights, np.array(hweights) * self.weights[i]))
        self.filters = filters
        self.weights = weights
        
        while len(self.filters) > self.max_filter_count:
            i, j = self.runnalls_kld_argmin()
            self.merge_filters(i, j)
        
        self.weights /= self.weights
        return [filter.x for filter in self.filters], [filter.P for filter in self.filters]

    def runnalls_kld_argmin(self):
        Dmin = float('Inf')
        ij = (None, None)
        for i in range(len(self.filters)):
            for j in range(i+1, len(self.filters)):
                Fi = self.filters[i]
                Fj = self.filters[j]
                wi = self.weights[i]
                wj = self.weights[j]

                Pij = Fi.P*(wi/(wi+wj)) + Fj.P*(wj/(wi+wj)) + np.dot(Fi.x-Fj.x,Fi.x-Fj.x)*(wi*wj/(wi+wj)**2)
                Dij = ((wi+wj)*np.log(np.linalg.det(Pij)) - wi*np.log(np.linalg.det(Fi.P)) - wj*np.log(np.linalg.det(Fj.P)))/2
                if Dij < Dmin:
                    Dmin = Dij
                    ij = (i, j)
        return ij

    def merge_filters(self, i, j):
        wi = self.weights[i]
        wj = self.weights[j]
        wij = wi + wj

        Fi = self.filters[i]
        Fj = self.filters[j]
        xij = (wi * Fi.x + wj * Fj.x) / wij
        Pij = (wi * Fi.P + wj * Fj.P + (wi*wj/wij) * np.dot(Fi.x - Fj.x, Fi.x - Fj.x)) / wij

        self.filters[i].x = xij
        self.filters[i].P = Pij
        self.weights[i] = wij
        del self.filters[j]
        self.weights = np.delete(self.weights, j)

    def map_estimate(self):
        max_idx = np.argmax(self.weights)
        map_filter = self.filters[max_idx]
        return map_filter.x, map_filter.P
    
    def mmse_estimate(self):
        means = np.array([filter.x for filter in self.filters])
        return means * self.weight
