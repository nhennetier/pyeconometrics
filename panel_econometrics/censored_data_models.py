# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st

import warnings
warnings.filterwarnings('ignore')

from numpy.linalg import inv
from math import exp, sqrt, log, pi

from panel_econometrics.base import CensoredBaseModel
from panel_econometrics.utils import inverse_mills_ratio, derivate_inverse_mills_ratio



class TobitModel(CensoredBaseModel):
    def __init__(self):
        self.name = 'Tobit I Model'

    def __response_function(self, X, beta):
        try:
            X.drop(self.output, axis=1, inplace=True)
        except:
            pass
        
        Z = 0
        for i,var in enumerate(self.variables):
            Z += beta[i] * X[var]

        return Z.rename('response')
        
    def __log_likelihood_censored(self, X, beta, sigma):
        Z = np.array(self.__response_function(X, beta))
        Z = Z/sigma

        norm_cdf_vec = np.vectorize(st.norm.cdf)
        result = np.sum(np.log(norm_cdf_vec(Z)))

        return result
            
    def __log_likelihood_uncensored(self, X, beta, sigma):
        Z = np.array(self.__response_function(X, beta))
        y = np.array(X[self.output])
        Z = 0.5 * np.multiply((y - Z)/sigma, (y - Z)/sigma)
        result = np.sum(Z)

        return result
            
    def __log_likelihood(self, X, beta, sigma):
        X_cens = X[X[self.output]==0]
        X_uncens = X[X[self.output]>0]

        result = - self.__log_likelihood_censored(X, beta, sigma) \
            - self.__log_likelihood_uncensored(X, beta, sigma) \
            - len(X_uncens) * log(sigma * sqrt(2*pi))

        return result
        
    def __grad_b_log_likelihood(self, X, b, s):
        X_cens = X[X[self.output]==0]
        X_uncens = X[X[self.output]>0]
        y_uncens = X_uncens[self.output]
        X_cens.drop(self.output, axis=1, inplace=True)
        X_uncens.drop(self.output, axis=1, inplace=True)

        inverse_mills_ratio_vec = np.vectorize(inverse_mills_ratio)

        grad_cens = inverse_mills_ratio_vec(-np.array(self.__response_function(X_cens, b), ndmin=2))
        grad_cens = - np.sum(np.multiply(X, grad_cens))

        grad_uncens = s * np.array(y_uncens) - np.array(self.__response_function(X_uncens, b), ndmin=2)
        grad_uncens = np.sum(np.multiply(X, grad_uncens))

        result = grad_cens + grad_uncens
        return result

    def __derivate_s_log_likelihood(self, X, b, s):
        X_uncens = X[X[self.output]>0]
        y_uncens = X_uncens[self.output]
        X_uncens.drop(self.output, axis=1, inplace=True)

        inverse_mills_ratio_vec = np.vectorize(inverse_mills_ratio)

        grad_uncens = s * np.array(y_uncens) - np.array(self.__response_function(X_uncens, b))
        grad_uncens = - np.sum(np.multiply(y_uncens, grad_uncens))

        result = grad_uncens + len(X_uncens)/s
        return result

    def __score(self, X, b, s):
        return np.concatenate([self.__grad_b_log_likelihood(X, b, s),
            self.__derivate_s_log_likelihood(X, b, s)])
            
    def __hessian_b_b(self, X, b, s):
        X_uncens = X[X[self.output]>0]
        y_uncens = X_uncens[self.output]
        X_uncens.drop(self.output, axis=1, inplace=True)

        derivate_inverse_mills_ratio_vec = np.vectorize(derivate_inverse_mills_ratio)
        hessian_uncens = 1 + derivate_inverse_mills_ratio_vec(-np.array(self.__response_function(X_uncens, b), ndmin=2))

        list_XXT = []
        for i in range(X_uncens.shape[0]):
            row = np.array(X_uncens[i,:], ndmin=2)
            list_XXT.append(row.T.dot(row))
        hessian_uncens = [-hessian_uncens[0,i]*list_XXT[i] for i in range(len(list_XXT))]
        
        result = sum(hessian_uncens)
        return result
        

    def __hessian_s_s(self, X, b, s):
        X_uncens = X[X[self.output]>0]
        y_uncens = X_uncens[self.output]

        item1 = -np.multiply(np.array(y_uncens), np.array(y_uncens))
        item2 = -1/s**2 
        result = np.sum(item1 + item2)
        return result

    def __hessian_b_s(self, X, b, s):
        X_uncens = X[X[self.output]>0]
        y_uncens = X_uncens[self.output]

        result = np.sum(X * np.array(y_uncens, ndmin=2).T, axis=0)
        return result
    
    def __hessian(self, X, beta):
        a = self.__hessian_b_b(X, b, s)
        b = self.__hessian_b_s(X, b, s)
        c = self.__hessian_s_s(X, b, s)

        item1 = np.concatenate([a,np.array(b, ndmin=2).T], axis=1)
        item2 = np.array(np.concatenate([b.T, np.array(c, ndmin=1)]), ndmin=2)
        result = np.concatenate([item1, item2], axis=0)
        return result

    def fit(self, X, output, nb_iter=20, drop_na=True, fill_value=None, verbose=False):
        '''Maximum Likelihhod Estimation
        Implement a Newton-Raphson algorithm to estimate parameters

        Parameters:
        ----------
        X: Dataframe
            Database to fit the model

        output: string
            Name of the variable to predict

        nb_iter: integer (optional, default 20)
            Maximal number of iteration before the end of the Newton-Raphson algorithm

        drop_na: boolean (optional, default True)
            Indicate the method to handle missing values in X
            If drop_na = False, fill_value has to be given

        fill_value: string or dict (optional, defaul None)
            Considered only if drop_na = False
            Possible values:
                - 'mean': missing values of a column are replaced by the mean of that column
                - 'median': missing values of a column are replaced by the median of that column
                - dict: keys must be variables' names and associated values the values used to fill Nan

        verbose: boolean (optional, default False)
            If set to True, allows prints of Newton-Raphson algorithm's progress
        '''
        self.output = output
        X = self.input_data_preparation(X.copy(), self.output, drop_na, fill_value)

        self.nb_censored_obs = len(X[X[self.output == 0]])
        self.nb_uncensored_obs = len(X[X[self.output > 0]])

        self.variables = [x for x in X.columns if x != self.output]
        
        beta_init = [0 for _ in range(len(self.variables))] + [1]   
        self.beta_est = np.zeros((nb_iter,len(beta_init)))
        self.beta_est[0] = beta_init

        X = X.groupby(level=0)

        self.init_ll = self.__log_likelihood(X, beta_init)

        if verbose:
            print('Initial log-likelihood : '+ str(self.init_ll))
            print('Parameters estimation in progress.')
        
        current_ll = self.init_ll
        prev_ll = self.init_ll
        j = 1
        while (j < nb_iter) \
            and (j == 1 or (current_ll - prev_ll > 0.01)):
            b = self.beta_est[j-1,:-1]/self.beta_est[j-1,-1]
            s = 1/self.beta_est[j-1,-1]
            
            score = self.__score(X, b, s)
            hessian = self.__hessian(X, b, s)

            try:
                step = inv(hessian).dot(score)
            except:
                raise ValueError('Improper classification problem' \
                    + ', should be 2 different labels')

            b += step[:-1]
            s += step[-1]
            self.beta_est[j] = np.concatenate([b, np.array(1/s, ndmin=2)])

            prev_ll = current_ll
            current_ll = self.__log_likelihood(X, self.beta_est[j,:-1],
                self.beta_est[j,-1])
            if verbose:              
                print('Iteration %s, log_likelihood : %s'\
                    % (j, current_ll))
            j += 1


        self.beta = self.beta_est[j-2,:-1]
        self.sigma = 
        self.beta_est = self.beta_est[:j-1,:]

        sqrt_vec = np.vectorize(sqrt)
        hessian = self.__hessian(X, self.beta_est[j-2])
        self.beta_se = sqrt_vec(inv(hessian).diagonal())

        self.confidence_interval = np.array(
                [[self.beta[i] - st.norm.ppf(0.975) * self.beta_se[i],
                    self.beta[i] + st.norm.ppf(0.975) * self.beta_se[i]]
                    for i in range(len(self.beta))])

        self.final_ll = prev_ll

        if j < nb_iter:
            self.converged = True
        else:
            self.converged = False

        return self