# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st
import scipy.integrate as spint

import warnings
warnings.filterwarnings('ignore')

from numpy.linalg import inv
from math import exp, sqrt, log

from panel_econometrics.base import PanelBaseModel
from panel_econometrics.utils import nCr, unique_permutations


class FixedEffectPanelModel(PanelBaseModel):
    '''Fixed Effects Logit model for Panel Data
    Estimation of parameters with the Conditional Maximum Likelihood method
    '''
    def __init__(self):
        self.name = 'Panel Fixed Effects Logit'

    def response_function(self, X, beta):
        try:
            X.drop(self.output, axis=1, inplace=True)
        except:
            pass
        
        Z = 0
        for i,var in enumerate(self.variables):
            Z += beta[i] * X[var]

        return Z.rename('response')
        
    def log_likelihood_obs(self, X, y, beta):
        X.reset_index(drop=True,inplace=True)
        y.reset_index(drop=True,inplace=True)

        Z = np.array(self.response_function(X, beta))

        if nCr(len(y),sum(y)) <= 100:
            perms = unique_permutations(y)
        else:
            perms = [np.random.permutation(y) for _ in range(100)]

        result = []
        for a in perms:
            result.append(np.exp(Z.dot(a)))

        result = Z.dot(np.array(y)) - log(sum(result))
        return result
            
    def log_likelihood(self, X, beta):
        result = sum(np.array(X.apply(lambda group : \
            self.log_likelihood_obs(group,
            group[self.output], beta))))

        return result
        
    def conditional_probability(self, X, y, beta):
        if nCr(len(y),sum(y)) <= 100:
            perms = unique_permutations(y)
        else:
            perms = [np.random.permutation(y) for _ in range(100)]

        result = []
        for z in perms:
            result.append(exp(np.array(z).T.dot(np.array(X).dot(beta))))

        result = np.sum(np.array(result), axis=0)
        result = exp(np.array(y).T.dot(np.array(X).dot(beta))) / result

        return result
    
    def score_obs(self, X, y, beta):
        X.drop(self.output, axis=1, inplace=True)

        X.reset_index(drop=True,inplace=True)
        y.reset_index(drop=True,inplace=True)

        if sum(y) == 0 or sum(y) == len(y):
            return np.array([0 for _ in range(len(X.columns))])

        else:
            if nCr(len(y),sum(y)) <= 100:
                perms = unique_permutations(y)
            else:
                perms = [np.random.permutation(y) for _ in range(100)]

            result = []
            for z in perms:
                result.append(np.array(z) \
                    * self.conditional_probability(X,z,beta))

            result = np.sum(np.array(result), axis=0)
            result = np.array(X).T.dot(np.array(y) - result)

            return result

    def score(self, X, beta):
        return np.sum(np.array(X.apply(lambda group : \
            self.score_obs(group, group[self.output], beta))), axis=0)
            
    def hessian_obs(self, X, y, beta):
        X.drop(self.output, axis=1, inplace=True)

        X.reset_index(drop=True,inplace=True)
        y.reset_index(drop=True,inplace=True)

        if sum(y) == 0 or sum(y) == len(y):
            return np.array([[0 for _ in range(len(X.columns))] \
                for _ in range(len(X.columns))])

        else:
            if nCr(len(y),sum(y)) <= 100:
                perms = unique_permutations(y)
            else:
                perms = [list(np.random.permutation(y)) for _ in range(100)]

            probas = []
            esp = []
            result = []
            i = 0
            for z in perms:
                probas.append(self.conditional_probability(X,z,beta))
                esp.append(np.array(z) * probas[i])
                result.append(np.array(z).dot(np.array(z).T) * probas[i])
                i += 1

            esp = np.sum(np.array(esp), axis=0)
            result = np.sum(np.array(result), axis=0)
            result = np.array(X).T.dot(
                result - esp.T.dot(esp)).dot(np.array(X))

            return -result

    def hessian(self, X, beta):
        return np.sum(np.array(X.apply(lambda group : \
            self.hessian_obs(group,group[self.output], beta))), axis=0)

    def fit(self, X, output, nb_iter=20, drop_na=True, fill_value=None, verbose=False):
        '''Maximum Likelihhod Estimation
        Implement a Newton-Raphson algorithm to estimate parameters

        Parameters:
        ----------
        X: 2-level MultiIndex Dataframe
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
        X = self.input_data_preparation(X.copy(), drop_na, fill_value)

        labels = list(np.unique(X[self.output]))
        if labels != [0,1]:
            raise ValueError("Labels must be in the unit interval.")
        
        self.nb_obs = len(X)
        self.variables = [x for x in X.columns if x!=self.output]
        
        beta_init = [0 for _ in range(len(self.variables))]   
        self.beta_est = np.zeros((nb_iter,len(beta_init)))
        self.beta_est[0] = beta_init

        X = X.groupby(level=0)

        self.init_ll = self.log_likelihood(X, beta_init)

        if verbose:
            print('Initial log-likelihood : '+ str(self.init_ll))
            print('Parameters estimation in progress.')
        
        j = 1
        while (j < nb_iter) and (j == 1 \
                or self.log_likelihood(X, self.beta_est[j-1]) \
                - self.log_likelihood(X, self.beta_est[j-2]) \
                > 0.01):
            
            score = self.score(X, self.beta_est[j-1])

            hessian = self.hessian(X, self.beta_est[j-1])

            try:
                self.beta_est[j] = self.beta_est[j-1] \
                    - inv(hessian).dot(score)
                if verbose:              
                    print('Iteration %s, log_likelihood : %s'\
                        % (j, self.log_likelihood(X, self.beta_est[j])))
                j += 1

            except:
                raise ValueError('Improper classification problem' \
                    + ', should be 2 different labels')

        self.beta = self.beta_est[j-2]
        self.beta_est = self.beta_est[:j-1,:]

        sqrt_vec = np.vectorize(sqrt)
        hessian = self.hessian(X, self.beta_est[j-2])
        self.beta_se = sqrt_vec(-inv(hessian).diagonal())

        self.confidence_interval = np.array(
                [[self.beta[i] - st.norm.ppf(0.975) * self.beta_se[i],
                    self.beta[i] + st.norm.ppf(0.975) * self.beta_se[i]]
                    for i in range(len(self.beta))])

        self.final_ll = self.log_likelihood(X, self.beta)

        if j < nb_iter:
            self.converged = True
        else:
            self.converged = False

        return self
        
    





class RandomEffectsPanelModel(PanelBaseModel):
    def __init__(self, residual_dist):
        self.name = 'Panel Random Effects Model'
        self.residual_dist = residual_dist

    def response_function(self, X, beta, mu):
        try:
            X.drop(self.output, axis=1, inplace=True)
        except:
            pass
        
        Z = mu
        for i,var in enumerate(self.variables):
            Z += beta[i] * X[var]

        return Z.rename('response')

    def calculus_tools(self, X, w, beta, mu, sigma):
        z = np.repeat(np.array([[1, w]]), X.shape[0], axis=0)
        z = np.concatenate((z, X), axis=1).T

        gamma = np.repeat(np.array([[mu, sigma]]), X.shape[0], axis=0)
        beta = np.array(beta, ndmin=2)
        beta = np.repeat(beta, X.shape[0], axis=0)
        gamma = np.concatenate((gamma, beta), axis=1).T

        return z, gamma

    def conditional_density_obs(self, X, w, y, beta, mu, sigma):
        z, gamma = self.calculus_tools(X, w, beta, mu, sigma)
        
        num = np.exp(np.multiply(np.array(y), z.T.dot(gamma)[:,0]))
        denom = 1 + np.exp(z.T.dot(gamma)[:,0])

        result = np.prod(np.divide(num, denom))

        return result

    def grad_conditional_density_obs(self, X, w, y, beta, mu, sigma):
        z, gamma = self.calculus_tools(X, w, beta, mu, sigma)

        item1 = np.exp(np.multiply(y, z.T.dot(gamma)[:,0]))
        item2 = np.exp(z.T.dot(gamma)[:,0])

        term1 = np.multiply(y, np.multiply(item1, 1+item2))
        term2 = np.multiply(item1, item2)
        denom = np.multiply(1+item2, 1+item2)

        result = np.divide(term1 - term2, denom)
        result = np.prod(result)
        result = np.multiply(result, np.prod(z, axis=1))

        return result
        
    def log_likelihood_obs(self, X, y, beta, mu, sigma):
        X.reset_index(drop=True,inplace=True)
        y.reset_index(drop=True,inplace=True)
        try:
            X.drop(self.output, axis=1, inplace=True)
        except:
            pass

        if self.residual_dist == 'probit':
            result = spint.quad(lambda w : self.conditional_density_obs(X, w, y, beta, mu, sigma) \
                * st.norm(0,1).pdf(w), -3*sigma, 3*sigma)[0]
        elif self.residual_dist == 'logit':
            result = spint.quad(lambda w : self.conditional_density_obs(X, w, y, beta, mu, sigma) \
                * st.logistic(0,1).pdf(w), -3*sigma, 3*sigma)[0]
        else:
            raise ValueError('Unknown value for argument residual_dist')
        
        return log(result)
        
            
    def log_likelihood(self, X, beta, mu, sigma):
        result = np.sum(np.array(X.apply(lambda group : \
            self.log_likelihood_obs(group, group[self.output], beta, mu, sigma))), axis=0)

        return result
        
    def score_obs(self, X, y, beta, mu, sigma):
        X.reset_index(drop=True,inplace=True)
        y.reset_index(drop=True,inplace=True)
        X.drop(self.output, axis=1, inplace=True)
        
        if self.residual_dist == 'probit':
            result = np.array([spint.quad(lambda w : self.grad_conditional_density_obs(X, w, y, beta, mu, sigma)[i] \
                * st.norm(0,1).pdf(w), -3*sigma, 3*sigma)[0] for i in range(len(beta)+2)])
        elif self.residual_dist == 'logit':
            result = np.array([spint.quad(lambda w : self.grad_conditional_density_obs(X, w, y, beta, mu, sigma)[i] \
                * st.logistic(0,1).pdf(w), -3*sigma, 3*sigma)[0] for i in range(len(beta)+2)])
        else:
            raise ValueError('Unknown value for argument residual_dist')

        result = result / exp(self.log_likelihood_obs(X, y, beta, mu, sigma))

        return result

    def score(self, X, beta, mu, sigma):
        return np.sum(np.array(X.apply(lambda group : \
            self.score_obs(group, group[self.output], beta, mu, sigma))), axis=0)
            
    def hessian(self, X, beta, mu, sigma):
        score_obs = X.apply(lambda group : np.array(self.score_obs(
            group, group[self.output], beta, mu, sigma), ndmin=2)).values
        score_obs = np.concatenate(list(score_obs))
        sum_score_obs = []
        for i in range(score_obs.shape[0]):
            row = np.array(score_obs[i,:], ndmin=2)
            sum_score_obs.append(row.T.dot(row))
        sum_score_obs = sum(sum_score_obs)
        print(sum_score_obs)

        score = np.array(self.score(X, beta, mu, sigma), ndmin=2).T
        print(score)

        result = sum_score_obs - score.dot(score.T) / len(X)
        print(result)
        
        return result

    def fit(self, X, output, nb_iter=20, drop_na=True, fill_value=None, verbose=False):
        self.output = output
        X = self.input_data_preparation(X.copy(), drop_na, fill_value)

        labels = list(np.unique(X[self.output]))
        if labels != [0,1]:
            raise ValueError("Labels must be in the unit interval.")
        
        self.nb_obs = len(X)
        self.variables = [x for x in X.columns if x!=self.output]
        
        beta_init = [0, 1] + [0 for _ in range(len(self.variables))]   
        self.beta_est = np.zeros((nb_iter,len(beta_init)))
        self.beta_est[0] = beta_init

        X = X.groupby(level=0)

        self.init_ll = self.log_likelihood(X, beta_init[2:], 0, 1)

        if verbose:
            print('Initial log-likelihood : '+ str(self.init_ll))
            print('Parameters estimation in progress.')
        
        j = 1
        while (j < nb_iter) and (j == 1 \
                or self.log_likelihood(X, self.beta_est[j-1,2:],
                    self.beta_est[j-1,0], self.beta_est[j-1,1]) \
                - self.log_likelihood(X, self.beta_est[j-2,2:],
                    self.beta_est[j-2,0], self.beta_est[j-2,1]) \
                > 0.01):
            
            score = self.score(X, self.beta_est[j-1,2:],
                self.beta_est[j-1,0], self.beta_est[j-1,1])

            hessian = self.hessian(X, self.beta_est[j-1,2:],
                self.beta_est[j-1,0], self.beta_est[j-1,1])

            try:
                self.beta_est[j] = self.beta_est[j-1] \
                    - inv(hessian).dot(score)
                if verbose:              
                    print('Iteration %s, log_likelihood : %s'\
                        % (j, self.log_likelihood(X, self.beta_est[j,2:],
                            self.beta_est[j,0], self.beta_est[j,1])))
                j += 1

            except:
                raise ValueError('Improper classification problem' \
                    + ', should be 2 different labels')

        self.beta = self.beta_est[j-2,2:]
        self.mu = self.beta_est[j-2,0]
        self.sigma = self.beta_est[j-2,1]
        self.beta_est = self.beta_est[:j-1,2:]

        sqrt_vec = np.vectorize(sqrt)
        hessian = self.hessian(X, self.beta_est[j-2,2:],
            self.beta_est[j-2,0], self.beta_est[j-2,1])
        self.beta_se = sqrt_vec(-inv(hessian).diagonal())

        self.confidence_interval = np.array(
                [[self.beta[i] - st.norm.ppf(0.975) * self.beta_se[i],
                    self.beta[i] + st.norm.ppf(0.975) * self.beta_se[i]]
                    for i in range(len(self.beta))])

        self.final_ll = self.log_likelihood(X, self.beta, self.mu, self.sigma)

        if j < nb_iter:
            self.converged = True
        else:
            self.converged = False

        return self