# Stochastic Processes in Finance

## Brownian Motion and Gaussian Processes

Standard Brownian motion $(W_t)$ satisfies:

- $W_0 = 0$
- independent increments
- Gaussian increments with mean 0 and variance $t-s$
- continuous paths

Geometric Brownian motion for an asset price $S_t$ is:

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t.
$$

This leads to a lognormal distribution for prices and is the foundation
of the Black-Scholes model.

## Itô's Lemma

Itô's lemma is the stochastic calculus analog of the chain rule. For a
smooth function $f(t, X_t)$,

$$
df = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW_t.
$$

This formula is essential for deriving option pricing equations and
stochastic integrals.

## Mean-Reverting Processes

The Ornstein-Uhlenbeck process is a mean-reverting diffusion:

$$
dX_t = \kappa(\theta - X_t) dt + \sigma dW_t.
$$

Mean reversion is commonly used to model interest rates, volatility,
and pairs-trading spreads.

## Jump Processes and Discontinuities

Pure diffusion models often understate tail risk. Jump-diffusion models
add discrete jumps:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t + S_{t-} dJ_t,
$$

where $J_t$ is a jump process such as a compound Poisson process.

## Risk-Neutral Valuation

Under a risk-neutral measure $Q$, discounted asset prices are martingales.
Option prices are computed as discounted expectations under $Q$ rather
than the physical probability measure $P$.

## Practical Takeaways

- Always identify the stochastic process you are assuming.
- Be explicit whether you are working under the physical or risk-neutral
  measure.
- Use mean-reversion and jump models only when empirical evidence
  supports them, and document the parameter estimation method.
