# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from numpy.linalg import inv
from math import exp, sqrt, log

from panel_econometrics.utils import nCr, norm_cdf, unique_permutations

    
  

class BaseModel():
    def input_data_preparation(self, X, drop_na=None, fill_value=None):
        try:
            X = X.to_frame()
        except:
            if len(X.index.names) != 2:
                raise ValueError("Only 2-level MultiIndex and Panel are supported.")

        if drop_na is not None:
            if drop_na:
                X.dropna(inplace=True)
            elif fill_value == 'mean':
                X.fillna(X.mean(), inplace=True)
            elif fill_value == 'median':
                X.fillna(X.median(), inplace=True)
            elif fill_value is not None:
                for var in X.columns:
                    try:
                        X[var].fillna(fill_value.get(var), inplace=True)
                    except:
                        raise ValueError('\'fill_value\' argument must be in list ' \
                            + '[\'mean\', \'median\'] or of type dict. See docstring for more info.')

        X.insert(0, 'const', 1)

        return X

    def fit(self, X, output, nb_iter=20, drop_na=True, fill_value=None, verbose=False):
        self.output = output
        X = self.input_data_preparation(X, drop_na, fill_value)

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
        
    def plot_trace_estimators(self):
        try:
            self.beta
        except:
            raise AttributeError('Fit method should be called before evaluating of the model')
            
        colors = ['b','g','r','c','m','y','k']
        for k in range(len(self.beta)):
            plt.plot(np.arange(1, len(self.beta_est)+1),
                     self.beta_est[:,k],
                     color=colors[(k-1) % len(colors)],
                     label="Beta_%s" % k)

        plt.xlim((1,len(self.beta_est)*1.2))
        plt.xlabel('Iterations')
        plt.ylabel('Estimators')
        plt.title('Trace plot of estimators of beta', size=16)
        plt.legend(loc='best')
        plt.show()

    def predict(self, X):
        try :
            self.beta
        except:
            raise AttributeError('Fit method should be called before evaluating the model.')

        X = self.input_data_preparation(X)

        Z = self.response_function(X, self.beta)
        result = (np.sign(Z)+1)/2

        try:
            result = result.astype(int).rename('predicted_label')
        except:
            raise ValueError('One or several data are missing.')

        return result
        
    def predict_proba(self, X):
        try :
            self.beta
        except:
            raise AttributeError('Fit method should be called before evaluating the model.')

        X = self.input_data_preparation(X)

        Z = self.response_function(X,self.beta)
        return Z.apply(lambda x : norm_cdf(x))

    def summary(self):
        try:
            self.beta
        except:
            raise AttributeError('Fit method should be called before evaluating of the model.')
            
        print('='*80)
        print('%-17s %30s %1s %-20s %8s' \
            % ('Dep. Variable:', self.output, ' ', 'Pseudo R-squ.:', 
                "%.5f" % (1-self.final_ll/self.init_ll)))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Model:', self.name, ' ', 'Log-Likelihood:', "%.3f" % self.final_ll))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Method:', 'MLE', ' ', 'LL-Null:', "%.3f" % self.init_ll))
        print('%-17s %30s %1s %-20s %8s' \
            % ('No. Observations:', self.nb_obs, ' ', 'LLR p-value:',
                "%.3f" % (1-st.chi2.cdf(2 * (self.final_ll - self.init_ll),
                len(self.beta)))))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Df Model:', len(self.beta)-1, ' ', ' ', ' '))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Converged:', self.converged, ' ', ' ', ' '))
        print('='*80)

        length = max([len(x) for x in self.variables if x != self.output])
        print('%25s %8s %8s %8s %8s %18s' \
            % (' ', 'coef', 'std err', 't', 'P>|t|', '[95.0% Conf. Int.]'))
        print('-'*80)

        for i,var in enumerate([x for x in self.variables if x != self.output]):
            print('%-24s %8s %8s %8s %8s %9s %9s' \
                % (var, 
                   "%.4f" % self.beta[i], 
                   "%.3f" % self.beta_se[i], 
                   "%.3f" % (self.beta[i] / self.beta_se[i]), 
                   "%.3f" % (1-st.norm.cdf(abs(self.beta[i]) / self.beta_se[i])), 
                   "%.3f" % self.confidence_interval[i, 0], 
                   "%.3f" % self.confidence_interval[i, 1])
                )
        print('='*80)





    
class FixedEffectPanelLogit(BaseModel):
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
        return sum(np.array(X.apply(lambda group : \
            self.score_obs(group, group[self.output],
            beta))))
            
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
        return sum(np.array(X.apply(lambda group : \
            self.hessian_obs(group,group[self.output],
            beta))))

        