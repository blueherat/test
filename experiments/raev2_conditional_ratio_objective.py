"""Convex conditional logistic head, caching its fixed linear term once."""
import numpy as np


class ConditionalRatioRisk:
    def __init__(self, positive, negative, signal, ridge):
        self.n, self.k, self.dim = positive.shape
        assert negative.shape == (self.n, self.dim) and signal.shape == (self.n,)
        assert np.all(signal > 0) and ridge >= 0
        self.positive = positive.reshape(self.n*self.k, self.dim)
        self.negative = negative
        self.signal = signal
        self.ridge = ridge
        self.linear = ((negative-positive.mean(1))/(4*signal[:, None])).mean(0)

    def __call__(self, weight):
        a = self.signal
        fp = (self.positive@weight).reshape(self.n, self.k)
        fq = self.negative@weight
        x, y = a[:, None]*fp/2, a*fq/2
        lp = np.logaddexp(x, -x)-np.log(2)
        lq = np.logaddexp(y, -y)-np.log(2)
        value = self.linear@weight + np.mean((lp.mean(1)+lq)/(2*a**2)) + self.ridge*(weight@weight)/2
        gradient = (self.linear
                    + self.positive.T@(np.tanh(x)/(4*a[:, None]*self.n*self.k)).reshape(-1)
                    + self.negative.T@(np.tanh(y)/(4*a*self.n)) + self.ridge*weight)
        return float(value), gradient

    def losses(self, weight):
        a = self.signal
        fp = (self.positive@weight).reshape(self.n, self.k)
        fq = self.negative@weight
        x, y = a[:, None]*fp/2, a*fq/2
        lp = np.logaddexp(x, -x)-np.log(2)
        lq = np.logaddexp(y, -y)-np.log(2)
        # Evaluate paired feature differences before dot products to retain the
        # high-noise shared-state cancellation in FP64.
        delta = self.negative-self.positive.reshape(self.n, self.k, self.dim).mean(1)
        return (delta@weight)/(4*a)+(lp.mean(1)+lq)/(2*a**2)
