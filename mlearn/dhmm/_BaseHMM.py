# -*- coding: utf-8 -*-

"""
Created on Jan 11, 2017

@author:Leon

This code is based on

 - Lawrence R. Rabiner, Proceedings of the IEEE 77.2, 257-286, 1989.
 - GuyZ's HMM implementation - https://github.com/guyz/HMM
"""

import numpy as np


class _BaseHMM(object):
    """
    Implements the basis for all deriving classes, but should not be used directly.
    """
    def __init__(self, n, m, precision=np.double, verbose=False):
        self.n = n
        self.m = m

        self.precision = precision
        self.verbose = verbose
        self._eta = self._eta1

    def _eta1(self, t, T):
        """
        Governs how each sample in the time series should be weighted.
        This is the default case where each sample has the same weight,
        i.e: this is a traditional HMM.
        """
        return 1.


    def forward_backward(self, observations, cache=False):
        """
        calculate the probability of the observation, given the model, i.e., P(O|model).
        alpha_t(i) = P(O1...Ot,qt=Si|model) - Conditional probability of state i and the observation up to time t.

        Returned: the log of the probability, i.e: the log likelihood model, give the observation - logL(model|O).

        In the discrete case, the value returned should be negative, since we are taking the log of actual (discrete)
        probabilities. In the continuous case, we are using PDFs which aren't normalized into actual probabilities,
        so the value could be positive.
        """
        if (cache is False):
            self._mapB(observations)

        alpha = self._calc_alpha(observations)
        return np.log(sum(alpha[-1]))

    def _calc_alpha(self, observations):
        """
        Calculates 'alpha' the forward variable.

        The alpha variable is a numpy array indexed by time, then state (TxN).
        alpha[t][i] = the probability of being in state 'i' after observing the
        first t symbols.
        """
        alpha = np.zeros((len(observations), self.n), dtype=self.precision)

        # init stage - alpha_1(i) = pi(i)b_i(O1)
        for i in range(self.n):
            alpha[0][i] = self.pi[i] * self.B_map[i][0]

        # induction
        for t in range(1, len(observations)):
            for j in range(self.n):
                for i in range(self.n):
                    alpha[t][j] += alpha[t - 1][i] * self.A[i][j]
                alpha[t][j] *= self.B_map[j][t]

        return alpha

    def _calc_beta(self, observations):
        """
        Calculates 'beta' the backward variable.

        The beta variable is a numpy array indexed by time, then state (TxN).
        beta[t][i] = the probability of being in state 'i' and then observing the
        symbols from t+1 to the end (T).
        """
        beta = np.zeros((len(observations), self.n), dtype=self.precision)

        # init stage
        for s in range(self.n):
            beta[len(observations) - 1][s] = 1.

        # induction
        for t in range(len(observations) - 2, -1, -1):
            for i in range(self.n):
                for j in range(self.n):
                    beta[t][i] += self.A[i][j] * self.B_map[j][t + 1] * beta[t + 1][j]

        return beta

    def decode(self, observations):
        """
        Find the best state sequence (path), given the model and an observation. i.e: max(P(Q|O,model)).

        This method is usually used to predict the next state after training.
        """
        # use Viterbi's algorithm. It is possible to add additional algorithms in the future.
        return self._viterbi(observations)

    def _viterbi(self, observations):
        """
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.

        delta[t][i] = max(P[q1...qt=Si,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.

        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1),
        i.e: the previous state.
        """
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        self._mapB(observations)

        delta = np.zeros((len(observations), self.n), dtype=self.precision)
        psi = np.zeros((len(observations), self.n), dtype=self.precision)

        # init
        for x in range(self.n):
            delta[0][x] = self.pi[x] * self.B_map[x][0]
            psi[0][x] = 0

        # induction
        for t in range(1, len(observations)):
            for j in range(self.n):
                for i in range(self.n):
                    if (delta[t][j] < delta[t - 1][i] * self.A[i][j]):
                        delta[t][j] = delta[t - 1][i] * self.A[i][j]
                        psi[t][j] = i
                delta[t][j] *= self.B_map[j][t]

        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = 0  # max value in time T (max)
        path = np.zeros((len(observations)), dtype=self.precision)
        for i in range(self.n):
            if (p_max < delta[len(observations) - 1][i]):
                p_max = delta[len(observations) - 1][i]
                path[len(observations) - 1] = i

                # path backtracing
                #        path = numpy.zeros((len(observations)),dtype=self.precision)
                # ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in range(1, len(observations)):
            path[len(observations) - i - 1] = psi[len(observations) - i][int(path[len(observations) - i])]
        return path

    def _calc_xi(self, observations, alpha=None, beta=None):
        """
        Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

        The xi variable is a numpy array indexed by time, state, and state (TxNxN).
        xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        time 't+1' given the entire observation sequence.
        """
        if alpha is None:
            alpha = self._calc_alpha(observations)
        if beta is None:
            beta = self._calc_beta(observations)
        xi = np.zeros((len(observations), self.n, self.n), dtype=self.precision)

        for t in range(len(observations) - 1):
            denom = 0.0
            for i in range(self.n):
                for j in range(self.n):
                    thing = 1.0
                    thing *= alpha[t][i]
                    thing *= self.A[i][j]
                    thing *= self.B_map[j][t + 1]
                    thing *= beta[t + 1][j]
                    denom += thing
            for i in range(self.n):
                for j in range(self.n):
                    numer = 1.0
                    numer *= alpha[t][i]
                    numer *= self.A[i][j]
                    numer *= self.B_map[j][t + 1]
                    numer *= beta[t + 1][j]
                    xi[t][i][j] = numer / denom

        return xi

    def _calc_gamma(self, xi, seqlen):
        """
        Calculates 'gamma' from xi.

        Gamma is a (TxN) numpy array, where gamma[t][i] = the probability of being
        in state 'i' at time 't' given the full observation sequence.
        """
        gamma = np.zeros((seqlen, self.n), dtype=self.precision)

        for t in range(seqlen):
            for i in range(self.n):
                gamma[t][i] = sum(xi[t][i])

        return gamma

    def train(self, observations, iterations=1, epsilon=0.0001, thres=-0.001):
        """
        Updates the HMMs parameters given a new set of observed sequences.

        observations can either be a single (1D) array of observed symbols, or when using
        a continuous HMM, a 2D array (matrix), where each row denotes a multivariate
        time sample (multiple features).

        Training is repeated 'iterations' times, or until log likelihood of the model
        increases by less than 'epsilon'.

        The 'thres' denotes the algorithms sensitivity to the log likelihood decreasing
        from one iteration to the other.
        """
        self._mapB(observations)

        for i in range(iterations):
            prob_old, prob_new = self.train_iter(observations)

            if (self.verbose):
                print("iter: ", i, ", L(model|O) =", prob_old, ", L(model_new|O) =", prob_new,
                      ", converging =", (prob_new - prob_old > thres))

            if (abs(prob_new - prob_old) < epsilon):
                # converged
                break

    def _update_model(self, new_model):
        """
        Replaces the current model parameters with the new ones.
        """
        self.pi = new_model['pi']
        self.A = new_model['A']

    def train_iter(self, observations):
        """
        A single iteration of an EM algorithm, which given the current HMM,
        computes new model parameters and internally replaces the old model
        with the new one.

        Returns the log likelihood of the old model (before the update),
        and the one for the new model.
        """
        # call the EM algorithm
        new_model = self._baum_welch(observations)

        # calculate the log likelihood of the previous model
        prob_old = self.forward_backward(observations, cache=True)

        # update the model with the new estimation
        self._update_model(new_model)

        # calculate the log likelihood of the new model.
        # Cache set to false in order to recompute probabilities of the observations give the model.
        prob_new = self.forward_backward(observations, cache=False)

        return prob_old, prob_new

    def _re_estimateA(self, observations, xi, gamma):
        """
        Re-estimation of the transition matrix (part of the 'M' step of Baum-Welch).
        Computes A_new = expected_transitions(i->j)/expected_transitions(i)

        Returns A_new, the modified transition matrix.
        """
        A_new = np.zeros((self.n, self.n), dtype=self.precision)
        for i in range(self.n):
            for j in range(self.n):
                numer = 0.0
                denom = 0.0
                for t in range(len(observations) - 1):
                    numer += (self._eta(t, len(observations) - 1) * xi[t][i][j])
                    denom += (self._eta(t, len(observations) - 1) * gamma[t][i])
                A_new[i][j] = numer / denom
        return A_new

    def _calc_stats(self, observations):
        """
        Calculates required statistics of the current model, as part
        of the Baum-Welch 'E' step.

        Deriving classes should override (extend) this method to include
        any additional computations their model requires.

        Returns 'stat's, a dictionary containing required statistics.
        """
        stats = {}

        stats['alpha'] = self._calc_alpha(observations)
        stats['beta'] = self._calc_beta(observations)
        stats['xi'] = self._calc_xi(observations, stats['alpha'], stats['beta'])
        stats['gamma'] = self._calc_gamma(stats['xi'], len(observations))

        return stats

    def _re_estimate(self, stats, observations):
        """
        Performs the 'M' step of the Baum-Welch algorithm.

        Deriving classes should override (extend) this method to include
        any additional computations their model requires.

        Returns 'new_model', a dictionary containing the new maximized
        model's parameters.
        """
        new_model = {}

        # new init vector is set to the frequency of being in each step at t=0
        new_model['pi'] = stats['gamma'][0]
        new_model['A'] = self._re_estimateA(observations, stats['xi'], stats['gamma'])

        return new_model

    def _baum_welch(self, observations):
        """
        An EM(expectation-modification) algorithm devised by Baum-Welch. Finds a local maximum
        that outputs the model that produces the highest probability, given a set of observations.

        Returns the new maximized model parameters
        """
        # E step - calculate statistics
        stats = self._calc_stats(observations)

        # M step
        return self._re_estimate(stats, observations)

    def _mapB(self, observations):
        """
        Deriving classes should implement this method, so that it maps the observations' mass/density Bj(Ot) to Bj(t).

        This method has no explicit return value, but it expects that 'self.B_map' is internally computed
        as mentioned above. 'self.B_map' is an (TxN) numpy array.

        The purpose of this method is to create a common parameter that will conform both to the discrete
        case where PMFs are used, and the continuous case where PDFs are used.

        For the continuous case, since PDFs of vectors could be computationally
        expensive (Matrix multiplications), this method also serves as a caching mechanism to significantly
        increase performance.
        """
        raise NotImplementedError("a mapping function for B(observable probabilities) must be implemented")
