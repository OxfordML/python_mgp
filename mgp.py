# -*- coding: utf-8 -*-
"""
An implementation of the approximate marginal GP.

Garnett, R., Osborne, M., and Hennig, P. Active Learning of Linear
Embeddings for Gaussian Processes. (2014). 30th Conference on
Uncertainty in Artificial Intelligence (UAI 2014).

Suppose we have a Gaussian process model on a latent function f:

  p(f | \theta) = GP(f; \mu(x; \theta), K(x, x'; \theta)),

where \theta are the hyperparameters of the model. Suppose we have a
dataset D = (X, y) of observations and a test point x*. This
function returns the mean and variance of the approximate marginal
predictive distributions for the associated observation value y* and
latent function value f*:

  p(y* | x*, D) = \int p(y* | x*, D, \theta) p(\theta | D) d\theta,
  p(f* | x*, D) = \int p(f* | x*, D, \theta) p(\theta | D) d\theta,

where we have marginalized over the hyperparameters \theta. The
approximate posterior is derived using he "MGP" approximation
described in the paper above.

Notes
-----

This code is only appropriate for GP regression! Exact inference
with a Gaussian observation likelihood is assumed.

This MGP implementation uses numerical gradients, in order to
allow any choice of kernel(s).

Dependencies
------------
- numpy
- GPy - https://github.com/SheffieldML/GPy
- numdifftools - https://github.com/pbrod/numdifftools


Usage
-----

Prediction with the MGP class work similar to GPy.models.GPRegression:

>>> gp = GPy.models.GPRegression(X, Y, kern=kernel)
>>> # Provide location of likelihood hps in model.param_array for now
>>> mgp = MGP(gp, lik_idx=np.array([-1]))
>>> mu_star, var_star = mgp.predict(x_star)

Inputs
------
gp - an instance of GPy.models.GPRegression. Should work with most kernels.
lik_idx - np.ndarray containing indexes of model.param_array
          corresponding to likelihood hps

Ahsan Alvi 2015
asa<at>robots.ox.ac.uk
"""

import numpy as np
import numdifftools as nd
import sys
import GPy


class MGP(object):

    def __init__(self, gp, lik_idx=None, verbose=None):
        """
        Defines all local variables of the MGP. Nomenclature follows
        the paper.
        :param gp: GPy.GP object whose hps are to be marginalised
        :param lik_idx: ndarray. Locations of the likelihood params
                        in the gp.param_array variable.
        """

        if verbose is None:
            self.verbose = False
        else:
            self.verbose = verbose

        self.gp = gp.copy()
        self.theta_backup = self.gp[:].copy()
        self.precision_matrix = None
        self.ind_flexible_params, self.ind_params_fixed = \
            find_free_hps(self.gp)
        self.logexp_params = None
        self._find_logexp_params()
        self._update_H()

        # Indexes of likelihood parameters. This is used to correct
        # the derivative of the posterior variance in
        # order to match the implementation based on GPML
        if lik_idx is None:
            if self.verbose:
                print 'Guessing index of the lik params in m.param_array'
            self.lik_idx = np.array([-1])
            # self._find_likelihood_params()
        else:
            self.lik_idx = lik_idx

    def _update_H(self, verbose=None):
        """
        Updates the posterior of the hyperparameters' Hessian
        """
        if verbose is not None:
            self.verbose = verbose

        if self.verbose:
            print "Computing Hessian of hyperparameter posterior...",
            sys.stdout.flush()

        # The precision_matrix is the negative of the Hessian of the
        # log likelihood. Garnett's Hessian in GPML is evaluated on the
        # *negative* log likelihood, so it isn't negated in his code.
        h = self._nd_gp_hessian()
        self.precision_matrix = -h

        if self.verbose:
            print "Done."
            print "Hessian:\n", h

    def predict(self, x_star, simple=None, delta_frac=None,
                debug=None):
        """
        Computes the quantities mhat, Vhat, their derivatives
        w.r.t. theta and then returns the
        MGP prediction at the locations x_star
        :param x_star: array of input locations to predict
        :param simple: boolean. If true, then the derivatives of
                       the mean and variance are found via a
                       simple central difference method instead
                       of using numdifftools. Default: True.
        :param delta_frac: float. Size of the gap between the
                           hyperparameters as a fraction of their
                           current values.
                           Optional and only used when simple=True.
                           Default = 0.01
        :param debug: bool. If true, then an extra dict is added
                      to the outputs containing intermediate
                      calculations

        :return: mean and variance at x_star as a tuple
        """
        if simple is None:
            simple = True
        if delta_frac is None:
            delta_frac = 0.01
        if debug is None:
            debug = False

        extras = {}  # extra outputs for debugging purposes

        # The MGP has the same mean as the parameterised GP
        param_gp_pred = self.gp.predict(x_star)
        mtilde = param_gp_pred[0].reshape((len(x_star), 1))
        Vhat = param_gp_pred[1].reshape((len(x_star), 1))
        # remove the likelihood variance, so that the prediction of
        # the *latent* function is inflated, not the observations.
        # This follows the matlab implememtation mgp.m line 241
        Vhat -= self.gp.likelihood.param_array
        Vtilde = np.zeros((len(x_star), 1))

        if self.verbose > 1:  # very verbose
            print "param_gp_pred", param_gp_pred
            print "mtilde", mtilde
            print "Vhat", Vhat
        dmhat_dtheta = np.zeros((len(x_star),
                                 len(self.ind_flexible_params)))
        dVhat_dtheta = np.zeros((len(x_star),
                                 len(self.ind_flexible_params)))

        if simple:
            # less accurate derivatives
            if self.verbose:
                print "Running fast prediction...",
            # 0 ------ 1 ------ 2
            # th-dth   th     th+dth
            delta_theta = (self.theta_backup[self.ind_flexible_params] *
                           (delta_frac))
            if self.verbose:
                print "theta:", self.theta_backup[self.ind_flexible_params]
                print "delta_theta:", delta_theta

            # Build the different thetas where each one has one
            # parameter varied by delta_theta.
            # This will be stored in a nested list, where the first idx
            # correponds to the theta we are varying and the second idx
            # will select the theta vector with either th-dth or th+dth
            # Only find differences for theta that are not fixed.
            new_thetas = []
            for ii in xrange(len(self.ind_flexible_params)):
                idx = self.ind_flexible_params[ii]
                th_minus = self.theta_backup.copy()
                th_minus[idx] = th_minus[idx] - delta_theta[ii]
                th_plus = self.theta_backup.copy()
                th_plus[idx] = th_plus[idx] + delta_theta[ii]
                new_thetas.append([th_minus, th_plus])

            if self.verbose > 1:  # very verbose
                print "new_thetas:", new_thetas

            # Same structure as new_thetas above
            mean_preds = []
            var_preds = []
            # Update the gp's hyperparameters and store the pred
            for ii in xrange(len(new_thetas)):
                # minus
                th_minus = new_thetas[ii][0]
                self.gp[:] = th_minus
                gp_pred = self.gp.predict(x_star)
                mean_minus, var_minus = (gp_pred[0][:, 0], gp_pred[1][:, 0])

                # plus
                th_plus = new_thetas[ii][1]
                self.gp[:] = th_plus
                gp_pred = self.gp.predict(x_star)
                mean_plus, var_plus = (gp_pred[0][:, 0], gp_pred[1][:, 0])

                # Store the results
                mean_preds.append([mean_minus, mean_plus])
                var_preds.append([var_minus, var_plus])

            self._reset_local_gp()

            if self.verbose > 1:  # very verbose
                print "mean_preds", mean_preds
                print "var_preds", var_preds

            # Once the mean and variance have been predicted at the
            # difference values of theta, find the derivatives
            # Each row corresponds to the theta, each column to x_star
            for ii in xrange(len(self.ind_flexible_params)):
                curr_dmhat_dtheta = ((mean_preds[ii][1]
                                     - mean_preds[ii][0])
                                     / (2*delta_theta[ii]))
                # print "mean_preds[ii][1]", mean_preds[ii][1].shape
                # print "mean_preds[ii][0]", mean_preds[ii][0].shape
                # print "curr_dmhat_dtheta", curr_dmhat_dtheta.shape
                dmhat_dtheta[:, ii] = curr_dmhat_dtheta
                curr_dVhat_dtheta = ((var_preds[ii][1]
                                      - var_preds[ii][0])
                                     / (2*delta_theta[ii]))
                dVhat_dtheta[:, ii] = curr_dVhat_dtheta

            # Correcting the derivatives w.r.t. likelihood terms
            for idx in self.lik_idx:
                dVhat_dtheta[:, idx] -= 1

            if self.verbose > 1:  # very verbose
                print "dmhat_dtheta", dmhat_dtheta
                print "dVhat_dtheta", dVhat_dtheta

            Vtilde = ((4. / 3.) * Vhat
                      + np.diag(dmhat_dtheta.dot(
                                np.linalg.solve(self.precision_matrix,
                                                dmhat_dtheta.T)))[:, None]
                      + np.diag(dVhat_dtheta.dot(
                                np.linalg.solve(self.precision_matrix,
                                                dVhat_dtheta.T)))[:, None]
                      / (3. * Vhat))
            if self.verbose:
                print "Vtilde", Vtilde

            if self.verbose:
                print "done."
        else:
            # More accurate gradients
            raise(NotImplementedError)

        # Add on the likelihood variance so that this prediction
        # is for the observations again.
        # TODO: how will this work in the multi-output case?
        Vtilde += self.gp.likelihood.param_array

        if debug:
            extras['delta_theta'] = delta_theta
            extras['new_thetas'] = new_thetas
            extras['theta_backup'] = self.theta_backup[
                self.ind_flexible_params]
            extras['mean_preds'] = mean_preds
            extras['var_preds'] = var_preds
            extras['Vhat'] = Vhat
            extras['dmhat_dtheta'] = dmhat_dtheta
            extras['dVhat_dtheta'] = dVhat_dtheta
            extras['precision_matrix'] = self.precision_matrix
            return mtilde, Vtilde, extras
        else:
            return mtilde, Vtilde

    def _reset_local_gp(self):
        """
        Sets the hyperparameters of self.gp to its original values.
        This is needed so that all the numerical derivatives are the same
        and don't rely on the previous evaluation
        """
        self.gp[:] = self.theta_backup.copy()

    def _nd_gp_hessian(self, step_max=None):
        """
        Numerically evaluated the Hessian matrix for all parameters
        of the GP that are not fixed.
        :return: 2D numpy array size len(theta)^2 corresponding to
                 the Hessian
        """
        if step_max is None:
            step_max = 0.1

        # Setting the GP params to the initial values, as they are
        # changed in-place for the different numerical computations
        self._reset_local_gp()

        # Find all the free hyperparameters
        # ind_flexible_params, ind_params_fixed = find_free_hps(self.gp)

        # TODO: convert theta to logexp where relevant
        theta = self.gp.param_array[self.ind_flexible_params]

        def ll_update(m, params, idx_theta):
            """
            Function that updates the GP's params and then returns
            the new log likelihood
            """
            # TODO: Deal with mixed logexp and untransformed hps
            temp = m[:]
            temp[idx_theta] = params
            m[:] = temp
            return m.log_likelihood()

        def ll_func(t):
            """
            Used by numdifftools
            """
            return ll_update(self.gp, t, self.ind_flexible_params)

        # TODO: if all hps are logexp, don't set step_nom or step_max
        H_func = nd.Hessian(ll_func, step_nom=theta, step_max=step_max)
        H = H_func(theta)

        # Setting the parameters to their original values again
        self._reset_local_gp()

        return H

    def _find_likelihood_params(self):
        """
        Find indexes of likelihood parameters and store them in
        self.lik_idx. This is used to correct the derivative of
        the posterior variance.
        """
        errtxt = 'Automatically finding likelihood params not implemented.' +\
                 ' Provide them for now.'
        raise NotImplementedError(errtxt)

    def _find_logexp_params(self):
        """
        Finds all the parameters that are constrained to by positive
        and updates self.logexp_params with their indexes.

        The hessian and derivatives of these parameters will be evaluated
        in log space and then transformed back into Cartesian space
        """

        # The indexes related to logexp parameters are in the tuple
        # with first element equal to sig:
        sig = GPy.core.parameterization.transformations.Logexp._instance
        for con_tuple in self.gp.constraints.items():
            if con_tuple[0] == sig:
                # con_tuple = (Logexp, array([indexes]))
                self.logexp_params = con_tuple[1]

    def transform_theta(self, theta):
        """
        Transforms an input vector (assumed to be theta).
        Hyperparameters that are constrained to be positive are
        transformed into log space.

        returns: transformed theta vector
        """
        # All hps are positive
        if np.sum(self.logexp_params) == len(self.theta_backup):
            return np.log(theta)
        elif np.sum(self.logexp_params) == 0:  # No positive hps
            return theta
        # mixed case
        else:
            tmp = np.log(theta[self.logexp_params])
            theta[self.logexp_params] = tmp
            return theta

    def inverse_transform_theta(self, theta):
        """
        Inverts the Logexp transform from the given theta vector.
        Hyperparameters that are constrained to be positive are
        transformed back.

        returns: untransformed theta vector
        """
        # All hps are positive
        if np.sum(self.logexp_params) == len(self.theta_backup):
            return np.exp(theta)
        elif np.sum(self.logexp_params) == 0:  # No positive hps
            return theta
        # mixed case
        else:
            tmp = np.exp(theta[self.logexp_params])
            theta[self.logexp_params] = tmp
            return theta


def find_free_hps(gp):
    """
    Returns locations of the hyperparameters that are not fixed as
    indexes of the GP's param_Array vector
    :param gp:
    :return:
    """
    # Find all the free hyperparameters
    ind_params_fixed = gp.constraints['fixed']
    temp_idx = np.arange(len(gp[:]))
    ind_flexible_params = np.delete(temp_idx, ind_params_fixed)
    return ind_flexible_params, ind_params_fixed


if __name__ == "__main__":
    pass
