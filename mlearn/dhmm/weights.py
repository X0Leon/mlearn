# -*- coding: utf-8 -*-

"""
Created on Jan 11, 2017

@author:Leon
"""

class EWMA(object):
    """
    Provides the ability to weigh samples in a time series as a function time.

    This mixin provides an exponentially weighted moving averages
    weighing function, that can be used to implement time-dependant HMMs.
    """

    def __init__(self):
        self.k = 20.0
        self.rho = 2 / (self.k + 1)
        self._eta = self._etaf

    def _etaf(self, t, T):
        """
        Exponentially Weighted Moving Averages weighing function, based on
        a paper published by Jonsson, 2008
        - t - current time sample
        - T - overall sequence length, 0-based index expected

        Note: valid when k < t <= T
        """
        return self.rho * ((1 - self.rho) ** (T - t))