# -*- coding: utf-8 -*-

"""
Created on Jan 13, 2017

@author:Leon Zhang
"""

import numpy as np

from .GMMHMM import GMMHMM
from .EMAGMMHMM import EMAGMMHMM


class DynamicHMM(object):
    """
    Dynamic HMM application - Gaussian mixture HMM with dynamic training pool

    Model attributes of GMMHMM or EMAGMMHMM:
    - n            number of hidden states
    - m            number of mixtures in each state (each 'symbol' like in the discrete case points to a mixture)
    - d            number of features (an observation can contain multiple features)
    - A            hidden states transition probability matrix ([NxN] np array)
    - means        means of the different mixtures ([NxMxD] np array)
    - covars       covars of the different mixtures ([NxM] array of [DxD] covar matrices)
    - w            weighing of each state's mixture components ([NxM] np array)
    - pi           initial state's PMF ([N] np array).

    Additional attributes:
    - min_std      used to create a covariance prior to prevent the covariances matrices from float underflow
    - precision    a np element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning

    Training attributes:
    - obs          Observations with d features (1D or 2D np array)
    - window       the window length of the training set, -1 for the whole observation sequence
    - multi_start  To overcome the local optimal of HMM, multiple random start may be useful

    Method:
    - train        Training HMM with dynamic pool using add-drop method
    """
    def __init__(self, n=5, m=4, d=2, A=None, means=None, covars=None, w=None, pi=None,
                 min_std=0.01, init_type='random', precision=np.double, verbose=False,
                 obs=None, model='GMMHMM', window=-1, multi_start=1):

        # Initialize the model parameters
        self.n, self.m, self.d,  self.obs, self.window, self.multi_start = n, m, d, obs, window, multi_start
        self.verbose, self.model = verbose, model
        self.init_models = []

        # Prepare several HMM models
        for _ in range(self.multi_start):
            if init_type == 'random':
                A, means, covars, w, pi = self._random_param()

            # Construct a HMM model
            if model == 'GMMHMM':
                hmm = GMMHMM(n, m, d, A, means, covars, w, pi, min_std, init_type, precision, verbose)
            else:
                hmm = EMAGMMHMM(n, m, d, A, means, covars, w, pi, min_std, init_type, precision, verbose)

            self.init_models.append(hmm)

            if init_type != 'random' and multi_start > 1:
                print("Error: use 'random' model init for multiple starts!")

    def train(self, iterations=1000, epsilon=0.0001, thres=-0.001):
        """
        Training HMM with dynamic pool using add-drop method
        """
        if self.obs is None:
            self.obs = self._random_obs()

        if self.window != -1 and self.window < len(self.obs):
            subset = self.obs[:self.window]
        else:
            subset = self.obs

        # Find best model to resist the local optimal
        scores = []
        for i,hmm in enumerate(self.init_models):
            hmm.train(subset, iterations, epsilon, thres)
            score = hmm.forward_backward(subset)
            print('Score: ', score)
            scores.append(score)

        hmm = self.init_models[scores.index(max(scores))]
        self.pi, self.A, self.w, self.means, self.covars = hmm.pi, hmm.A, hmm.w, hmm.means, hmm.covars
        print('Hidden states', hmm.decode(subset))
        predict_obs = hmm.predict(subset)
        print('Predict next observation: ', predict_obs)

        # dynamic training
        if self.window != -1 and self.window < len(self.obs):
            for new_o in self.obs[self.window:]:
                subset[0:self.window-1] = subset[1:self.window]
                subset[-1] = new_o

                if self.model == 'GMMHMM':
                    hmm = GMMHMM(self.n, self.m, self.d, self.A, self.means, self.covars, self.w, self.pi,
                                 min_std=0.01, init_type='update', precision=np.double, verbose=self.verbose)
                else:
                    hmm = EMAGMMHMM(self.n, self.m, self.d, self.A, self.means, self.covars, self.w, self.pi,
                                    min_std=0.01, init_type='update', precision=np.double, verbose=self.verbose)
                hmm.train(subset, iterations, epsilon, thres)
                self.pi, self.A, self.w, self.means, self.covars = hmm.pi, hmm.A, hmm.w, hmm.means, hmm.covars
                predict_obs = hmm.predict(subset)
                print('Predict next observation: ', predict_obs)

    def _random_obs(self):
        """
        Generate random data set as observations
        """
        obs = np.array((0.6 * np.random.random_sample((40, self.d)) - 0.3), dtype=np.double)
        return obs

    def _random_param(self):
        """
        Generate random mode parameters for beginners
        """
        a = np.random.random_sample((self.n, self.n))
        row_sums = a.sum(axis=1)
        a = np.array(a / row_sums[:, np.newaxis], dtype=np.double)

        w = np.random.random_sample((self.n, self.m))
        row_sums = w.sum(axis=1)
        w = np.array(w / row_sums[:, np.newaxis], dtype=np.double)

        means = np.array((0.6 * np.random.random_sample((self.n, self.m, self.d)) - 0.3), dtype=np.double)
        covars = [[np.matrix(np.eye(self.d, self.d)) for _ in range(self.m)] for _ in range(self.n)]

        pi = np.random.random_sample((self.n))
        pi = np.array(pi / sum(pi), dtype=np.double)

        return a, means, covars, w, pi
