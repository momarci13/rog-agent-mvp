# Markov Chains and Hidden Markov Models

## Markov Property

A Markov chain is a stochastic process $(X_t)$ with the property that
future states depend only on the present state, not on the full history:

$$
P(X_{t+1} = j \mid X_t = i, X_{t-1}, \ldots, X_0) = P(X_{t+1} = j \mid X_t = i).
$$

This property simplifies analysis and is widely used in finance,
statistics, and signal processing.

## Transition Matrices

For a finite state space, the transition matrix $P$ has entries
$P_{ij} = P(X_{t+1} = j \mid X_t = i)$.
A stationary distribution $\pi$ satisfies

$$
\pi = \pi P,
$$

and $\pi$ gives the long-run fraction of time spent in each state.

## Ergodicity and Mixing

A chain is ergodic if it has a unique stationary distribution and
forgets its starting state over time. Ergodicity implies that empirical
averages converge to expectation under $\pi$.

## First-Passage and Hitting Times

First-passage time measures how long it takes to reach a target state
for the first time. In finance, this is analogous to drawdown-triggered
exits or barrier option hitting times.

## Hidden Markov Models (HMMs)

In an HMM, an unobserved state process $(Z_t)$ evolves as a Markov chain
and observations $(Y_t)$ are generated conditionally on $Z_t$.
HMMs are used for regime-switching models, credit ratings, and
volatility states.

The common HMM inference tasks are:

- Filtering: estimate $P(Z_t \mid Y_{1:t})$.
- Smoothing: estimate $P(Z_t \mid Y_{1:T})$.
- Decoding: find the most likely state sequence.

### Applications in Finance

- Regime-switching return models
- Credit rating migration matrices
- Discrete-time Markov chains for portfolio state transitions
- Monte Carlo simulation of latent regimes

### Practical Notes

- Always test the Markov assumption. If state persistence is too weak,
  the chain may not be a useful model.
- Estimate transition matrices with enough data to avoid overfitting.
- If probabilities are estimated from counts, use shrinkage or
  regularization to avoid zero-probability states.
