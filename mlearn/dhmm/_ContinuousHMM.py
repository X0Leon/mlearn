# -*- coding: utf-8 -*-

"""
Created on Jan 11, 2017

@author:Leon
"""

from ._BaseHMM import _BaseHMM
import numpy as np


class _ContinuousHMM(_BaseHMM):
    """
    A Continuous HMM - This is a base class implementation for HMMs with
    mixtures. A mixture is a weighted sum of several continuous distributions,
    which can therefore create a more flexible general PDF for each hidden state.

    This class can be derived, but should not be used directly. Deriving classes
    should generally only implement the PDF function of the mixtures.

    Model attributes:
    - n            number of hidden states
    - m            number of mixtures in each state (each 'symbol' like in the discrete case points to a mixture)
    - d            number of features (an observation can contain multiple features)
    - A            hidden states transition probability matrix ([NxN] np array)
    - means        means of the different mixtures ([NxMxD] np array)
    - covars       covars of the different mixtures ([NxM] array of [DxD] covar matrices)
    - w            weighing of each state's mixture components ([NxM] np array)
    - pi           initial state's PMF ([N] np array).

    Additional attributes:
    - min_std      used to create a covariance prior to prevent the covariances matrices from underflowing
    - precision    a np element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning
    """

    def __init__(self, n, m, d=1, A=None, means=None, covars=None, w=None, pi=None, min_std=0.01, init_type='uniform',
                 precision=np.double, verbose=False):
        """
        Construct a new Continuous HMM.
        In order to initialize the model with custom parameters,
        pass values for (A,means,covars,w,pi), and set the init_type to 'user'.

        Normal initialization uses a uniform distribution for all probabilities, and is not recommended.
        """
        _BaseHMM.__init__(self, n, m, precision, verbose)  # @UndefinedVariable

        self.d = d
        self.A = A
        self.pi = pi
        self.means = means
        self.covars = covars
        self.w = w
        self.min_std = min_std

        self.reset(init_type=init_type)

    def reset(self, init_type='uniform'):
        """
        If required, initialize the model parameters according the selected policy
        """
        if init_type == 'uniform':
            self.pi = np.ones((self.n), dtype=self.precision) * (1.0 / self.n)
            self.A = np.ones((self.n, self.n), dtype=self.precision) * (1.0 / self.n)
            self.w = np.ones((self.n, self.m), dtype=self.precision) * (1.0 / self.m)
            self.means = np.zeros((self.n, self.m, self.d), dtype=self.precision)
            self.covars = [[np.matrix(np.ones((self.d, self.d), dtype=self.precision)) for j in range(self.m)]
                           for i in range(self.n)]
        elif init_type == 'user':
            # if the user provided a 4-d array as the covars, replace it with a 2-d array of np matrices.
            covars_tmp = [[np.matrix(np.ones((self.d, self.d), dtype=self.precision)) for j in range(self.m)] for
                          i in range(self.n)]
            for i in range(self.n):
                for j in range(self.m):
                    if type(self.covars[i][j]) is np.ndarray:
                        covars_tmp[i][j] = np.matrix(self.covars[i][j])
                    else:
                        covars_tmp[i][j] = self.covars[i][j]
            self.covars = covars_tmp

    def _mapB(self, observations):
        """
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        This method highly optimizes the running time, since all PDF calculations
        are done here once in each training iteration.

        - self.Bmix_map - computes and maps Bjm(Ot) to Bjm(t).
        """
        self.B_map = np.zeros((self.n, len(observations)), dtype=self.precision)
        self.Bmix_map = np.zeros((self.n, self.m, len(observations)), dtype=self.precision)
        for j in range(self.n):
            for t in range(len(observations)):
                self.B_map[j][t] = self._calc_bjt(j, t, observations[t])

    """
    b[j][Ot] = sum(1...M)w[j][m]*b[j][m][Ot]
    Returns b[j][Ot] based on the current model parameters (means, covars, weights) for the mixtures.
    - j - state
    - Ot - the current observation
    Note: there's no need to get the observation itself as it has been used for calculation before.
    """

    def _calc_bjt(self, j, t, Ot):
        """
        Helper method to compute Bj(Ot) = sum(1...M){Wjm*Bjm(Ot)}
        """
        bjt = 0
        for m in range(self.m):
            self.Bmix_map[j][m][t] = self._pdf(Ot, self.means[j][m], self.covars[j][m])
            bjt += (self.w[j][m] * self.Bmix_map[j][m][t])
        return bjt

    def _calc_gammamix(self, alpha, beta, observations):
        """
        Calculates 'gamma_mix'.

        Gamma_mix is a (TxNxK) np array, where gamma_mix[t][i][m] = the probability of being
        in state 'i' at time 't' with mixture 'm' given the full observation sequence.
        """
        gamma_mix = np.zeros((len(observations), self.n, self.m), dtype=self.precision)

        for t in range(len(observations)):
            for j in range(self.n):
                for m in range(self.m):
                    alphabeta = 0.0
                    for jj in range(self.n):
                        alphabeta += alpha[t][jj] * beta[t][jj]
                    comp1 = (alpha[t][j] * beta[t][j]) / alphabeta

                    bjk_sum = 0.0
                    for k in range(self.m):
                        bjk_sum += (self.w[j][k] * self.Bmix_map[j][k][t])
                    comp2 = (self.w[j][m] * self.Bmix_map[j][m][t]) / bjk_sum

                    gamma_mix[t][j][m] = comp1 * comp2

        return gamma_mix

    def _update_model(self, new_model):
        """
        Required extension of _update_model. Adds 'w', 'means', 'covars',
        which holds the in-state information. Specifically, the parameters
        of the different mixtures.
        """
        _BaseHMM._update_model(self, new_model)  # @UndefinedVariable

        self.w = new_model['w']
        self.means = new_model['means']
        self.covars = new_model['covars']

    def _calc_stats(self, observations):
        """
        Extension of the original method so that it includes the computation
        of 'gamma_mix' stat.
        """
        stats = _BaseHMM._calc_stats(self, observations)  # @UndefinedVariable
        stats['gamma_mix'] = self._calc_gammamix(stats['alpha'], stats['beta'], observations)

        return stats

    def _re_estimate(self, stats, observations):
        """
        Required extension of _re_estimate.
        Adds a re-estimation of the mixture parameters 'w', 'means', 'covars'.
        """
        # re-estimate A, pi
        new_model = _BaseHMM._re_estimate(self, stats, observations)  # @UndefinedVariable

        # re-estimate the continuous probability parameters of the mixtures
        w_new, means_new, covars_new = self._re_estimateMixtures(observations, stats['gamma_mix'])

        new_model['w'] = w_new
        new_model['means'] = means_new
        new_model['covars'] = covars_new

        return new_model

    def _re_estimateMixtures(self, observations, gamma_mix):
        """
        Helper method that performs the Baum-Welch 'M' step
        for the mixture parameters - 'w', 'means', 'covars'.
        """
        w_new = np.zeros((self.n, self.m), dtype=self.precision)
        means_new = np.zeros((self.n, self.m, self.d), dtype=self.precision)
        covars_new = [[np.matrix(np.zeros((self.d, self.d), dtype=self.precision)) for j in range(self.m)] for i
                      in range(self.n)]

        for j in range(self.n):
            for m in range(self.m):
                numer = 0.0
                denom = 0.0
                for t in range(len(observations)):
                    for k in range(self.m):
                        denom += (self._eta(t, len(observations) - 1) * gamma_mix[t][j][k])
                    numer += (self._eta(t, len(observations) - 1) * gamma_mix[t][j][m])
                w_new[j][m] = numer / denom
            w_new[j] = self._normalize(w_new[j])

        for j in range(self.n):
            for m in range(self.m):
                numer = np.zeros((self.d), dtype=self.precision)
                denom = np.zeros((self.d), dtype=self.precision)
                for t in range(len(observations)):
                    numer += (self._eta(t, len(observations) - 1) * gamma_mix[t][j][m] * observations[t])
                    denom += (self._eta(t, len(observations) - 1) * gamma_mix[t][j][m])
                means_new[j][m] = numer / denom

        cov_prior = [[np.matrix(self.min_std * np.eye((self.d), dtype=self.precision)) for j in range(self.m)]
                     for i in range(self.n)]
        for j in range(self.n):
            for m in range(self.m):
                numer = np.matrix(np.zeros((self.d, self.d), dtype=self.precision))
                denom = np.matrix(np.zeros((self.d, self.d), dtype=self.precision))
                for t in range(len(observations)):
                    vector_as_mat = np.matrix((observations[t] - self.means[j][m]), dtype=self.precision)
                    numer += (self._eta(t, len(observations) - 1) * gamma_mix[t][j][m] * np.dot(vector_as_mat.T,
                                                                                                   vector_as_mat))
                    denom += (self._eta(t, len(observations) - 1) * gamma_mix[t][j][m])
                covars_new[j][m] = numer / denom
                covars_new[j][m] = covars_new[j][m] + cov_prior[j][m]

        return w_new, means_new, covars_new

    def _normalize(self, arr):
        """
        Helper method to normalize probabilities, so that
        they all sum to '1'
        """
        summ = np.sum(arr)
        for i in range(len(arr)):
            arr[i] = (arr[i] / summ)
        return arr

    def _pdf(self, x, mean, covar):
        """
        Deriving classes should implement this method. This is the specific
        Probability Distribution Function that will be used in each
        mixture component.
        """
        raise NotImplementedError("PDF function must be implemented")