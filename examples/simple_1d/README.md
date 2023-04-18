# Simple 1d examples
We have several examples in this directory. In all these examples, out data samples are 1-dimensional.

## Uniform distribution
We train a score function estimator to approximates the score of a uniform distribution
```
p(x) = 1/b if a <= x <= a+b
p(x) = 0 otherwise.
```
We then generate samples using Langevin dynamics.
