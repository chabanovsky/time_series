# encoding:utf-8
import csv
import collections
import math
import copy
import logging
from operator import itemgetter
from datetime import datetime, timedelta
import numpy as np


class STLDecoposition:
    def __init__(self, initial_ts, no=2, ni=10, np=7, nl=7, ns=5, nt=7):
        self.initial_ts = initial_ts
        self.np = np
        self.ni = ni
        self.no = no
        self.nl = nl
        self.ns = ns
        self.nt = nt
        # It's interesting that if we use polynomial of degree 2 the residuals became really small.
        # # It looks like it's better to use linear approximation
        self.d = 2

        self.siries = [ (day, number, day.weekday()) for day, number in self.initial_ts ]
        
        self.ts_values  = [ number for _, number in self.initial_ts ]
        self.residuals  = [0 for _ in range(len(self.siries))]
        self.seasonal   = [0 for _ in range(len(self.siries))]
        self.trend      = [0 for _ in range(len(self.siries))]

        self.ros = [1 for _ in range(len(self.siries))]
        self.vs =  [0 for _ in range(len(self.siries))]

    def decompose(self):
        for outer_index in range(0, self.no):
            for inner_index in range(0, self.ni):
                self.do_inner_loop()
            self.do_outer_loop()

    def seasonal_ts(self):
        return [(day, self.seasonal[index]) for index, (day, _) in enumerate(self.initial_ts)]

    def trend_ts(self):
        return [(day, self.trend[index]) for index, (day, _) in enumerate(self.initial_ts)]

    def residuals_ts(self):
        return [(day, self.residuals[index]) for index, (day, _) in enumerate(self.initial_ts)]

    def do_outer_loop(self):
        self.residuals = self.get_residuals()
        for index in range(len(self.residuals)):
            self.ros[index] = self.ro(index)
    
    def do_inner_loop(self):
        # Step 1
        detrended = self.detrend(self.ts_values)
        # Step 2
        subseries = self.extract_weekly_subseries(detrended)
        seasonal_by_weekday = dict()
        for index, series in enumerate(subseries):
            seasonal_by_weekday[index] = self.loess_smoothing(series, self.ns, self.d)

        cycle_subseries = self.back_to_seasonal_ts(seasonal_by_weekday)
        # Step 3
        # for us self.np must be 7, since we work with weekly data
        l_of_cycle = self.moving_average(cycle_subseries, self.np, True)
        l_of_cycle = self.moving_average(l_of_cycle, self.np, True)
        l_of_cycle = self.moving_average(l_of_cycle, 3, True)
        l_of_cycle = self.loess_smoothing(l_of_cycle, self.nl, self.d)
            
        # Step 4
        self.seasonal = np.subtract(cycle_subseries, l_of_cycle).tolist()
        # Step 5
        deseasonaled = self.deseasonal(self.ts_values)
        # Step 6
        self.trend = self.loess_smoothing(deseasonaled, self.nt, self.d)

    def loess_smoothing(self, values, q, d):
        result = list()
        values_lenght = len(values)
        for index_in_series in range(0, values_lenght):
            start, end = self.get_neighborhoods_indexes(values_lenght, index_in_series, q)
            loess_weights = [self.v(values, index_in_series, second_lvl_index) for second_lvl_index in range(start, end+1)]
            loess_series = values[start: end+1]
            loess_item_index = index_in_series - start
            x = [index for index in range(0, len(loess_series))]
            loess_result = np.polyfit(x, loess_series, d)
            if d == 1:
                y = x[loess_item_index] * loess_result[0] + loess_result[1]
            if d == 2:
                y = pow(x[loess_item_index], 2) * loess_result[0] + x[loess_item_index] * loess_result[1] + loess_result[2]
            result.append(y)        

        return result

    def back_to_seasonal_ts(self, seasonal_by_weekday):
        result = list()
        indexes = [0 for _ in range(0, self.np)]
        for day, _ in self.initial_ts:
            weekday = day.weekday()
            index_in_weekday_series = indexes[weekday]
            item = seasonal_by_weekday[weekday][index_in_weekday_series]
            indexes[weekday] += 1
            result.append(item)

        return result
        
                
    def get_neighborhoods_indexes(self, lenght, index, q):
        if lenght < q:
            return 0, lenght-1

        m = 1 if q % 2 == 0 else 0
        start = index - q/2 
        end = index + q/2 - m
        
        if start < 0:
            end += abs(start)
            start = 0
        if end >= lenght:
            start -= (end - lenght + 1)
            end = lenght - 1

        return start, end

    def get_residuals(self):
        return self.detrend(self.deseasonal(self.ts_values))

    def extract_weekly_subseries(self, values):
        def build(vals, weekday_index):
            return [ vals[index] for index, (day, _) in enumerate(self.initial_ts) if day.weekday() == weekday_index]

        monday = build(values, 0)
        tuesday = build(values, 1)
        wednesday = build(values, 2)
        thursday = build(values, 3)
        friday = build(values, 4)
        saturday = build(values, 5)
        sunday = build(values, 6)

        return [monday, tuesday, wednesday, thursday, friday, saturday, sunday]

    def detrend(self, values):
        return np.subtract(values, self.trend).tolist()

    def deseasonal(self, values):
        return np.subtract(values, self.seasonal).tolist()

    def ro(self, index):
        h = 6 * np.median(self.residuals)
        if h == 0:
            return 0
        u = abs(self.residuals[index])/h
        return self.B(u)
    
    def v(self, values, x_index, element_index):
        if x_index == element_index:
            return 1
        x = values[x_index]
        xi = values[element_index]
        u = abs(x - xi)/(max(x_index, element_index) - min(x_index, element_index))
        return self.W(u)

    def W(self, u):
        if u >= 0 and u < 1:
            return pow((1-pow(u, 3)), 3)
        return 0

    def B(self, u):
        if u >= 0 and u < 1:
            return pow((1-pow(u, 2)), 2)
        return 0


    def moving_average(self, values, n=3, fix_size=False) :
        result = np.cumsum(values, dtype=float)
        result[n:] = result[n:] - result[:-n]
        result = result[n - 1:] / n

        if not fix_size:
            return result.tolist()

        n = n-1
        to_start = (n / 2) + (n % 2)
        to_end = (n / 2)
        for _ in range (0, to_start):
            result = np.insert(result, 0, result[0])

        for _ in range (0, to_end):
            lenght = len(result)
            result = np.insert(result, lenght, result[lenght-1])            

        return result.tolist()

