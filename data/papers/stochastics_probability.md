# Stochastics and Probability Theory

## Probability Fundamentals

Probability theory provides the mathematical foundation for uncertainty.
A probability space is defined by a triple $(\Omega, \mathcal{F}, P)$
where $\Omega$ is the sample space, $\mathcal{F}$ is a sigma-algebra of
events, and $P$ is a probability measure with $P(\Omega) = 1$.

A random variable $X$ is a measurable mapping from $\Omega$ to the real
numbers. Its distribution is given by the cumulative distribution
function $F_X(x) = P(X \le x)$.

### Expectation and Variance

The expectation of $X$ is

$$
\mathbb{E}[X] = \int_{\Omega} X(\omega) \, dP(\omega).
$$

For discrete variables, $\mathbb{E}[X] = \sum_x x P(X = x)$. The variance
measures dispersion:

$$
\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2.
$$

### Independence and Conditional Probability

Two events $A$ and $B$ are independent if $P(A \cap B) = P(A)P(B)$.
For random variables, independence implies joint density factorization.
Conditional probability is defined as

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0.
$$

Bayes' theorem is the fundamental identity:

$$
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.
$$

### Law of Large Numbers and Central Limit Theorem

The law of large numbers states that the sample average converges to the
true expectation as the number of observations grows. The central limit
theorem says that the normalized sum of independent random variables
tends toward a Gaussian distribution under mild conditions.

These results justify statistical inference and explain why many random
quantities appear approximately normal.

### Stochastic Processes

A stochastic process is a sequence of random variables $(X_t)_{t\ge 0}$.
Important examples include random walks, Poisson processes, and
Brownian motion.

A process is a martingale if

$$
\mathbb{E}[X_{t+1} \mid \mathcal{F}_t] = X_t,
$$

which expresses a fair-game property. Martingale methods are central to
modern financial math and to analyzing stopping times.

### Distributions Common in Quantitative Finance

- Normal / Gaussian: useful for returns approximations and CLT-based
  inference.
- Lognormal: common for asset prices under geometric Brownian motion.
- Student-
  t: heavier tails than Gaussian, often used for return residuals.
- Exponential / Poisson: models waiting times and jump arrivals.

### Practical Takeaways

- Always state distributional assumptions explicitly: normality, IID,
  stationarity, finite variance.
- Use probability laws to justify why a test statistic or risk measure is
  meaningful.
- Understand when approximation results (CLT, law of large numbers)
  apply and when they fail.
