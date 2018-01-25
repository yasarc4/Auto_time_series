import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import math
from collections import deque
import datetime as dt
from sklearn.metrics import mean_squared_error,mean_absolute_error

class ThreshDetector(object):

    def __init__(self, threshold=0.2, window_length=10, min_training=30):
        self._window = deque(maxlen=window_length)
        self._threshold = threshold
        self._triggered = False
        self.changepoint = 0
        self._min_training = min_training
        self.changepoints = []
        self._changepoint_stats = {}
        self._sum = 0
        self._sumsq = 0
        self._N = 0

    def reset(self):
        self.changepoints.append(self.changepoint)
        self._changepoint_stats[self.changepoint]={'n':self._N*1,'sum':self._sum*1.0,'sumsq':self._sumsq*1.0}
        self._sum = 0
        self._sumsq = 0
        self._N = 0
        self._triggered=False

    def step(self, datum):
        self._window.append(datum)
        self._N += 1
        self._sum += datum
        self._sumsq += datum ** 2
        self._mu = self._sum / self._N
        if self._N > self._min_training:
            variance = (self._sumsq - self._N * self._mu ** 2) / (self._N - 1)
            self._std = math.sqrt(variance)
            window_mu = sum(self._window) / len(self._window)
            ratio = window_mu / self._mu # TODO: Will fail if mu is zero.
            if ratio > (1.0 + self._threshold) or ratio < (1.0 - self._threshold):
                self._triggered = True
                self.changepoint += self._N
                self.reset()
                return True
        return self._triggered
