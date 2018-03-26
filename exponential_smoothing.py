# encoding:utf-8
import csv
import collections
import math
import copy
import logging
from operator import itemgetter
from datetime import datetime, timedelta
import numpy as np

class ExponentialSmoothing:
    def __init__(self, initial_ts, model_starts_at=0, alpha=0.1, beta=0.05, gamma=0.8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.initial_ts = initial_ts
        self.model_starts_at = model_starts_at
        self.seasonal_period = 7 # weakly

        if self.model_starts_at == 0:
            self.model_starts_at = len(self.initial_ts) / 2
        self.working_ts = [number if index < self.model_starts_at else 0 for index, (day, number) in enumerate(self.initial_ts)]

    def weighted_average(self, back_in_time=50):
        lenght = len(self.initial_ts)
        for index in range(self.model_starts_at, lenght):
            if back_in_time == 0:
                self.working_ts[index] = self.alpha * self.working_ts[index-1] + (1 - self.alpha) * self.working_ts[index-2]
            else:
                self.working_ts[index] = 0
                for pow_index in range(0, back_in_time):
                    self.working_ts[index] += self.alpha * pow((1 - self.alpha), pow_index) * self.working_ts[index-pow_index-1]
        
        return [(day, self.working_ts[index]) for index, (day, number) in enumerate(self.initial_ts)]

    def average(self):
        data = [number for _, number in self.initial_ts]
        return [(day, np.mean(data[:index+1])) for index, (day, _) in enumerate(self.initial_ts)]

    def weighted_smoothing(self):
        lenght = len(self.initial_ts)
        result = [self.initial_ts[0][1]]
        for index in range(1, lenght):
            result.append(self.alpha * float(self.initial_ts[index][1]) + (1 - self.alpha) * float(self.initial_ts[index-1][1]))
        
        return [(day, result[index]) for index, (day, number) in enumerate(self.initial_ts)]

    def holt_winter_helper(self, series):
        lenght = len(series)
        result = [number if index < self.model_starts_at else 0 for index, (day, number) in enumerate(series)]

        initial_level = np.mean(result[:self.model_starts_at])
        levels = [initial_level if index < self.model_starts_at else 0 for index in range(0, lenght)]

        b_ts = [result[index+self.seasonal_period]-result[index] for index in range(0, self.model_starts_at-self.seasonal_period)]
        initial_trend = np.mean(b_ts)/self.seasonal_period
        trends = [initial_trend if index < self.model_starts_at else 0 for index in range(0, lenght)]
        seasonals = [result[index] - levels[index] if index < self.model_starts_at else 0 for index in range(0, lenght)]

        for index in range(self.model_starts_at-1, lenght-1):
            levels[index] = self.alpha * (result[index] - seasonals[index-self.seasonal_period]) + (1 - self.alpha) * (levels[index-1] + trends[index-1])
            trends[index] = self.beta * (levels[index] - levels[index-1]) + (1 - self.beta) * trends[index-1]
            seasonals[index] = self.gamma * (result[index] - levels[index]) + (1 - self.gamma) * seasonals[index-self.seasonal_period]

            result[index+1] = levels[index] + trends[index] + seasonals[index+1-self.seasonal_period]

        return [(day, result[index]) for index, (day, number) in enumerate(series)] 

    def holt_winter(self):
        return self.holt_winter_helper(self.initial_ts)

    def hw_on_stl(self, stl_seasonal, stl_trend, stl_residuals):
        seasonal = self.holt_winter_helper(stl_seasonal)
        trend = self.holt_winter_helper(stl_trend)
        residuals = self.holt_winter_helper(stl_residuals)

        lenght = len(stl_seasonal)
        return [(stl_seasonal[index][0], seasonal[index][1] + trend[index][1] + residuals[index][1]) for index in range(0, lenght)]
    