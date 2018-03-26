# encoding:utf-8
import csv
import collections
import math
import copy
import logging
from operator import itemgetter
from datetime import datetime, timedelta
import numpy as np
from stl_decompose import STLDecoposition

class SSA:
    def __init__(self, initial_ts, model_starts_at=0):
        self.initial_ts = initial_ts
        self.lenght = len(self.initial_ts)
        self.model_starts_at = model_starts_at
        if self.model_starts_at == 0:
            self.model_starts_at = self.lenght/2

    def embeddings(self, series, tau):
        lenght = len(series)
        values = [number for day, number in series]

        result = np.zeros((tau, lenght-tau), dtype=float)
        for index in range (0, lenght-tau):
            result[:,index] = values[index:index+tau]

        return result

    def covariance_matrix(self, np_matrix):
        m, n = np_matrix.shape
        C = 1./float(n) * (np_matrix.dot(np_matrix.transpose()))
        return C

    def ssa(self):
        result = [(day, float(number)) for day, number in self.initial_ts]
        current_end = self.lenght - 10 # self.model_starts_at
        components = None
        
        while current_end < self.lenght:
            current_series = [(day, number) for index, (day, number) in enumerate(result) if index < current_end]
            current_tau = int((current_end+1)/2)

            new_value, recreated_series, components = self.ssa_helper(current_series, current_tau)
            result[current_end] = (result[current_end][0], new_value)
            current_end +=1

        return recreated_series, components
    
    def ssa_helper(self, series, tau):
        X = self.embeddings(series, tau)
        m, n = X.shape
        C = self.covariance_matrix(X)
        eigvals, eigvecs = np.linalg.eigh(C)

        lambda_ = sorted(eigvals, reverse=True)
        ev_m, ev_n = eigvecs.shape

        U = np.zeros((ev_m, ev_n), dtype=float)
        u_index = 0
        for index in range(ev_n-1, -1, -1):
            U[:,u_index] = eigvecs[:, index]
            u_index += 1
    
        # Filter out lambdas less then zero
        positive_lambdas = list()
        positive_eigvecs = list()
        for index, value in enumerate(lambda_):
            if value <= 0:
                break
            positive_lambdas.append(value)
            positive_eigvecs.append(U[:,index].tolist())

        V = list()
        for index in range(0, len(positive_lambdas)):
            vec = np.zeros((len(positive_eigvecs[index]), 1), dtype=float)
            vec[:,0] = positive_eigvecs[index] 
            v_i = X.transpose().dot(vec)/math.sqrt(positive_lambdas[index])
            V.append(v_i[:,0].tolist())

        Xs = list()
        for index in range(0, len(positive_lambdas)):
            vec = np.zeros((len(positive_eigvecs[index]), 1), dtype=float)
            vec[:,0] = positive_eigvecs[index]
            v_vec = np.zeros((1, len(V[index])), dtype=float)
            v_vec[0] = V[index]

            xs_i = vec.dot(v_vec) * math.sqrt(positive_lambdas[index])
            Xs.append(xs_i)
            
        # Now we need to split Xs by different groups
        splited_Xs = dict()
        first = 0
        second = 0
        last = len(Xs)

        normed_lambdas = positive_lambdas/np.linalg.norm(positive_lambdas)
        while second < last:
            if abs(math.log(positive_lambdas[first]/positive_lambdas[second], 10)) < 1:
                if splited_Xs.get(first, None) is None:
                    splited_Xs[first] = list()
                splited_Xs[first].append(Xs[second])
                second +=1
            else:
                #logging.error("Added from: %s to: %s // abs(log(%s / %s))= %s" % (str(first), str(second-1), str(positive_lambdas[first]), str(positive_lambdas[second]), str( abs(math.log(positive_lambdas[first]/positive_lambdas[second], 10)) )  ) )
                first = second

        #for index in sorted(splited_Xs):
        #    logging.error("Xs lenghts: index %s, lenght %s" % (str(index), str(len(splited_Xs[index]))))

        def get_a_series(Xs_local, days):
            X_hat = np.zeros((Xs_local[0].shape), dtype=float)
            for index in range(0, len(Xs_local)):
                X_hat = np.add(X_hat, Xs_local[index])

            hat_m, hat_n = X_hat.shape
            result_values = list()
            for index in range(0, hat_m + hat_n - 1):
                result_values.append(list())

            for index in range(0, hat_n):
                for inner_index in range(index, index + hat_m):
                    result_values[inner_index].append(X_hat[inner_index-index][index])
                    #logging.error("Adding to %s element[%s][%s] of ([%s][%s])" % (str(inner_index), str(inner_index-index), str(index), str(hat_m), str(hat_n)))

            result = [float(np.mean(values)) for values in result_values]
            return [(days[index], value) for index, value in enumerate(result)]

        days = [day for day, number in series]
        
        splited_series = list()
        for index in sorted(splited_Xs):
            splited_series.append(get_a_series(splited_Xs[index], days))

        recreated_series = list()
        for main_index in range(0, len(splited_series[0])):
            a_day = splited_series[0][main_index][0]
            a_value = splited_series[0][main_index][1]
            values = list()
            for sub_index in range(1, len(splited_series)):
                values.append(splited_series[sub_index][main_index][1])
            recreated_series.append((a_day, a_value + sum(values)))

        # Calculate a new value    
        def make_prediction(eigvecs_to_work_with, Q):
            eigvec_tau_values = [value[len(value)-1] for value in eigvecs_to_work_with]
            eigvec_tau = np.zeros((1, len(eigvec_tau_values)), dtype=float)
            eigvec_tau[0] = eigvec_tau_values

            eigvecs_asterisk = np.array([value[:-1] for value in eigvecs_to_work_with])
            new_value = eigvec_tau.dot(eigvecs_asterisk)
            v_sqr = float(eigvec_tau.dot(eigvec_tau.transpose()))
            if v_sqr == 1.:                
                return 0.
            new_value =  new_value / float(1. - v_sqr)
            #logging.error("Value %s, shape %s, v_sqr %s" % (str(new_value), str(new_value.shape), str(1./float(1. - v_sqr)) ) )
            predicted = float(new_value.dot(Q)[0][0])

            return predicted

        Q_values = [number for day, number in series[len(series)-tau+1:]]
        Q = np.zeros((len(Q_values), 1))
        Q[:,0] = Q_values

#        while np.linalg.matrix_rank(np.array(positive_eigvecs)) >= tau and len(positive_eigvecs) > 1:
#            positive_eigvecs = positive_eigvecs[:-1]
#        predicted = make_prediction(positive_eigvecs, Q)
#        return predicted, recreated_series, splited_series     

        eigvec_indexes = sorted(splited_Xs)
        eigvec_indexes.append(len(positive_eigvecs))
        predictions = list()
        for index in range(1, 2): # len(eigvec_indexes)-3):
            from_index = eigvec_indexes[index]
            to_index = eigvec_indexes[index+1]
            #logging.error("Indexes are 0 + %s - %s" % (str(from_index), str(to_index)))
            eigvecs_to_work_with = [positive_eigvecs[0]]
            eigvecs_to_work_with.extend(positive_eigvecs[from_index:to_index])
            predictions.append(make_prediction(eigvecs_to_work_with, Q)) 

        predicted = float(np.mean(predictions))

#        logging.error("Rnak %s, vecs %s" %  (str(np.linalg.matrix_rank(np.array(positive_eigvecs).transpose())), str(len(positive_eigvecs))))
        return predicted, recreated_series, splited_series 



