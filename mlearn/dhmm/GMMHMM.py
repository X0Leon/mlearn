# -*- coding: utf-8 -*-

"""
Created on Jan 11, 2017

@author:Leon
"""
import numpy as np
from ._ContinuousHMM import _ContinuousHMM


class GMMHMM(_ContinuousHMM):
    """
    A Gaussian Mixtures HMM - This is a representation of a continuous HMM,
    containing a mixture of gaussian distributions in each hidden state.

    For more information, refer to _ContinuousHMM.
    """
    def __init__(self, n, m, d=1, A=None, means=None, covars=None, w=None, pi=None,
                 min_std=0.01, init_type='uniform', precision=np.double, verbose=False):
        """
        See _ContinuousHMM constructor for more information
        """
        _ContinuousHMM.__init__(self, n, m, d, A, means, covars, w, pi, min_std, init_type, precision,
                                verbose)  # @UndefinedVariable

    def _pdf(self, x, mean, covar):
        """
        Gaussian PDF function
        """
        covar_det = np.linalg.det(covar)

        c = (1 / ((2.0 * np.pi) ** (float(self.d / 2.0)) * (covar_det) ** (0.5)))
        pdfval = c * np.exp(-0.5 * np.dot(np.dot((x - mean), covar.I), (x - mean)))
        return pdfval
