# encoding:utf-8
import csv
import collections
import math
import copy
import logging
from operator import itemgetter
from datetime import datetime, timedelta
import numpy as np

class Autoregressive:
    def __init__(self, initial_ts, modeling=False, model_starts_at=0, period=7):
        self.modeling = modeling
        if self.modeling:
            lenght = len(initial_ts)
            starting_day = initial_ts[lenght-1][0]
            self.initial_ts = [(day, number) for day, number in initial_ts]
            self.initial_ts.extend( [ (starting_day+timedelta(days=index+1), 0) for index in range(0, model_starts_at)] )
            self.model_starts_at = lenght
            self.lenght = len(initial_ts)
        else:
            self.initial_ts = initial_ts
            self.lenght = len(self.initial_ts)
            self.model_starts_at = model_starts_at if model_starts_at > 0 else self.lenght/2
        self.seasonal_period = period

    def diff_helper(self, series, diff_degree=1, lag=1):
        diff_data = [(day, number) for day, number in series]
        for _ in range(0, diff_degree):
            diff_data = [ (diff_data[index][0], diff_data[index][1] - diff_data[index-lag][1] ) for index in range(lag, len(diff_data))]

        return diff_data

    def seasonal_diff(self, series, period):
        return self.diff_helper(series, 1, period)

    def stationary_helper(self, series, period):
        if period > 1:
            without_seasonal = self.seasonal_diff(series, period)
        else:
            without_seasonal = series

        stationary_series = self.diff_helper(without_seasonal, 1)

        to_add = len(series) - len(stationary_series)
        for index in range(to_add-1, -1, -1):
            stationary_series.insert(0, (series[index][0], 0))

        return stationary_series

    def stationary(self):
        return self.stationary_helper(self.initial_ts, self.seasonal_period)

    def get_ar_points(self, series, period=7, depth=1):
        X = list()
        Y = list()
        lenght = len(series)
        for index in range(period*depth, lenght):
            _, y_value = series[index]
            Y.append(y_value)
            x_values = list()
            for sub_index in range(index-period, index-period*(depth+1), -period):
                _, x_value = series[sub_index]
                x_values.append(x_value)
            X.append(x_values)        
        return Y, X
    
    @staticmethod
    def a_matrix(X, depth):
        x_lenght = len(X)
        A = np.empty((x_lenght, depth))
        #A[:, 0] = 1
        for index in range (0, x_lenght):
            for sub_index in range(0, depth):
                A[index][sub_index] = X[index][sub_index]
        return A

    def theta_from_linear_regression(self, series, period=7, depth=1):
        Y, X = self.get_ar_points(series, period, depth)
        # Y=A * theta = T_1 * 1 + T_2 * x
        A = Autoregressive.a_matrix(X, depth)

        theta = np.linalg.pinv(A).dot(Y)
        return theta

    def autoregressive(self):
        ar = self.autoregressive_helper(self.initial_ts, self.seasonal_period, 10)
        st = self.stationary_helper(ar, self.seasonal_period)
        to_add = len(ar) - len(st)
        for index in range(0, to_add):
            st.insert(0, (ar[index][0], 0))
        return [(day, number) if index < self.model_starts_at else (day, number + st[index][1]) for index, (day, number) in enumerate(ar)]

    def autoregressive_helper(self, series, period, depth):
        lenght = len(series)
        for_theta = [(day, number) for index, (day, number) in enumerate(series) if index < self.model_starts_at]
        result = [(day, number) if index < self.model_starts_at else (day, 0) for index, (day, number) in enumerate(series)]

        theta = self.theta_from_linear_regression(for_theta, period, depth)
        
        for index in range(self.model_starts_at, lenght):
            x_values = list()
            for seasonal_index in range(index-period, index-period*(depth+1), -period):
                _, value = result[seasonal_index]
                x_values.append(value)
            A = Autoregressive.a_matrix([x_values], depth)

            this_value = float(A.dot(theta))
            day, _ = result[index]
            result[index] = (day, this_value)

        return result

    def arima_helper(self, series, p, q, period):
        lenght = len(series)
        result = [(day, number) if index < self.model_starts_at else (day, 0) for index, (day, number) in enumerate(series)]

        stationary = self.stationary_helper(series, period)
        # y_hat = y_t - e_t => y_t = y_hat + e_t = theta_1 * y_1 + theta_2 * y_2 + theta_3 * e_t1 + theta_4 * e_t2 + e_t
        Y_values = [(day, number) for index, (day, number) in enumerate(series) if index < self.model_starts_at]
        E_values = [(day, number) for index, (day, number) in enumerate(stationary) if index < self.model_starts_at]
        
        Y, X = self.get_ar_points(Y_values, period, p)
        Y1, X2 = self.get_ar_points(E_values, period, q)
        
        for index in range (0, len(X)):
            X[index].extend(X2[index])

        A = Autoregressive.a_matrix(X, p+q)
        theta = np.linalg.pinv(A).dot(Y) 

        for index in range(self.model_starts_at, lenght):
            y_values = list()
            e_values = list()
            result_stationary = self.stationary_helper(result[0:self.model_starts_at+index], period)
            for seasonal_index in range(index-period, index-period*(p+1), -period):
                _, y_value = result[seasonal_index]
                y_values.append(y_value)
            for seasonal_index in range(index-period, index-period*(q+1), -period):    
                _, e_value = result_stationary[seasonal_index]
                e_values.append(e_value)

            values = list(y_values)
            values.extend(e_values)
            m = np.mean([result[mean_index][1] for mean_index in range(0, index)])
            c = m * (1 - sum([theta[theta_index] for theta_index in range(0, p)]))
            this_A = Autoregressive.a_matrix([values], p+q)

            this_value = float(this_A.dot(theta)) + c
            
            day, _ = result[index]
            result[index] = (day, this_value)

        return result

    def arima(self, lag=10):
        result = self.arima_helper(self.initial_ts, lag, lag, self.seasonal_period)
        if self.modeling:
            return result[self.model_starts_at:]
        return result
        
