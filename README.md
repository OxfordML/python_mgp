# python_mgp

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
