# -*- coding: utf-8 -*-

"""
Created on Jan 11, 2017

@author:Leon
"""
from .GMMHMM import GMMHMM
from .weights import EWMA
import numpy as np


class EMAGMMHMM(GMMHMM, EWMA):
    '''
    An Exponentially Weighted Moving Averages Gaussian Mixtures HMM -
    This is a representation of a continuous HMM, containing a mixture of gaussians in
    each hidden state, and includes an internal weighing function that gives more
    significance to newer observations.

    It should be noted that an open issue is that log likelihood fails to increase
    in each iteration using such weighted observations, although the equations
    appear to be sound. Nevertheless, this is the first open source implementation
    of such HMMs.

    For more information, refer to GMMHMM.
    '''

    def __init__(self, n, m, d=1, A=None, means=None, covars=None, w=None, pi=None,
                 min_std=0.01, init_type='uniform', precision=np.double, verbose=False):
        print("Warning: weighted EMs may not converge due to the log-likelihood function may decrease.")

        GMMHMM.__init__(self, n, m, d, A, means, covars, w, pi, min_std, init_type, precision, verbose)
        EWMA.__init__(self)