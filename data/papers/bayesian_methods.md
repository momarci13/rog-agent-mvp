# Bayesian Methods for Quantitative Analysis

## Bayesian Inference Basics

Bayesian inference updates beliefs using data. Starting from a prior
distribution $p(\theta)$ over parameters, the posterior is

$$
p(\theta \mid y) = \frac{p(y \mid \theta) p(\theta)}{p(y)},
$$

where $p(y \mid \theta)$ is the likelihood and $p(y)$ is the marginal
likelihood.

### Priors, Posteriors, and Conjugacy

A conjugate prior yields a posterior in the same family as the prior.
Example:

- Bernoulli likelihood with Beta prior gives a Beta posterior.
- Gaussian likelihood with known variance and Gaussian prior gives a
  Gaussian posterior.

Conjugacy allows analytical updating and rapid computation.

### Bayesian Decision Theory

Bayesian decision-making minimizes expected posterior loss.
For squared-error loss, the optimal point estimate is the posterior mean.
For asymmetric loss, choose the posterior quantile that minimizes risk.

### Credible Intervals vs Confidence Intervals

A credible interval $[a, b]$ satisfies

$$
P(\theta \in [a,b] \mid y) = 1 - \alpha.
$$

This is a probability statement about the parameter itself, unlike a
frequentist confidence interval.

### Hierarchical Models and Parameter Uncertainty

Hierarchical models introduce hyperparameters that capture group-level
variation. They are especially useful when data are scarce or noisy.

Bayesian treatment of parameter uncertainty is important in finance,
because point estimates for expected return and volatility are highly
unstable.

### Computational Methods

When analytic posteriors are unavailable, use:

- Markov chain Monte Carlo (MCMC)
- Gibbs sampling
- Metropolis-Hastings
- Variational inference

These methods approximate the posterior distribution rather than a
single point estimate.

### Practical Bayesian Principles

- Always specify the prior and justify it: is it informative,
  weakly informative, or noninformative?
- Make explicit the likelihood model and its assumptions.
- Report posterior distributions, not just posterior means.
- Use predictive checks and posterior predictive distributions to validate
your model.
