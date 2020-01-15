"""Class to optimize rounding on the regression output
"""
import sklearn.metrics as metrics
import scipy as sp
import functools as ft
import numpy as np


class OptimizedRounder(object):
    def __init__(self, default_coefficients=[0.5,
                                             1.5,
                                             2.5,
                                             3.0]):
        self.coefficients = default_coefficients

    def kappa_loss(self, coefficients, y1, y2):
        """kappa loss: a loss function that maps rounders onto a real number
            intuitively representing cost.  
            This function computes the opposite of Cohen's kappa statistic, 
            a score that expresses the level of agreement between two 
            annotators on a classification problem. It is defined as

            .. math::
                \kappa = (p_o - p_e) / (1 - p_e)

            where :math:`p_o` is the empirical probability of agreement on the 
            label assigned to any sample (the observed agreement ratio), and 
            :math:`p_e` is the expected agreement when both annotators assign 
            labels randomly. :math:`p_e` is estimated using a per-annotator 
            empirical prior over the class labels.

        Parameters
        ----------
        coefficients : array, shape = [4]
            Rounder coefficients.

        y1 : array, shape = [n_samples]
            Output of regression.

        y2 : array, shape = [n_samples]
            True labels.

        Returns
        -------
        kappa_loss : float
            the opposite of kappa statistic, which is a number between -1 and 1. The maximum
            value means complete agreement; zero or lower means chance agreement.

        References
        ----------
        .. [1] J. Cohen (1960). "A coefficient of agreement for nominal scales".
            Educational and Psychological Measurement 20(1):37-46.
            doi:10.1177/001316446002000104.
        .. [2] `R. Artstein and M. Poesio (2008). "Inter-coder agreement for
            computational linguistics". Computational Linguistics 34(4):555-596.
            <https://www.mitpressjournals.org/doi/pdf/10.1162/coli.07-034-R2>`_
        .. [3] `Wikipedia entry for the Cohen's kappa.
                <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_
        """
        labels = np.copy(y1)
        for i, pred in enumerate(labels):
            if pred < coefficients[0]:
                labels[i] = 0
            elif pred < coefficients[1]:
                labels[i] = 1
            elif pred < coefficients[2]:
                labels[i] = 2
            elif pred < coefficients[3]:
                labels[i] = 3
            else:
                labels[i] = 4
        ll = metrics.cohen_kappa_score(y2,
                                       labels,
                                       weights='quadratic')
        return -ll

    def fit(self, y1, y2):
        """
        OptimizedRounder fits rounding with coefficients w = (w1, …, w4) 
        to minimize the kappa loss function between the observed targets in 
        the dataset, and the targets predicted by regression.

        Parameters
        ----------
        y1 : array, shape = [n_samples]
            Output of regression.

        y2 : array, shape = [n_samples]
            True labels.
        """
        loss_partial = ft.partial(self.kappa_loss,
                                  y1=y1,
                                  y2=y2)
        self.coefficients = sp.optimize.minimize(fun=loss_partial, 
                                                 x0=self.coefficients,
                                                 method='nelder-mead')['x']

    def predict(self, y1, coefficients):
        """
        Predict using a selected rounder.

        Parameters
        ----------
        y1 : array, shape = [n_samples]
            Output of regression.
        coefficients : array, shape = [4]
            Rounder coefficients.

        Returns
        -------
        labels : array, shape = [n_samples]
            predicted label by the rounder.
        """
        labels = np.copy(y1)
        for i, pred in enumerate(labels):
            if pred < coefficients[0]:
                labels[i] = 0
            elif pred >= coefficients[0] and pred < coefficients[1]:
                labels[i] = 1
            elif pred >= coefficients[1] and pred < coefficients[2]:
                labels[i] = 2
            elif pred >= coefficients[2] and pred < coefficients[3]:
                labels[i] = 3
            else:
                labels[i] = 4
        return labels
