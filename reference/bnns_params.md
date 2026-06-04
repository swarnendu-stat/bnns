# Parameter functions for Bayesian Neural Networks

These functions provide `dials` parameter objects for tuning the
hyperparameters of Bayesian Neural Networks.

## Usage

``` r
L(range = c(1L, 5L), trans = NULL)

warmup(range = c(100L, 1000L), trans = NULL)

chains(range = c(1L, 4L), trans = NULL)

iter(range = c(500L, 2000L), trans = NULL)

nodes(range = c(1L, 64L), trans = NULL)

act_fn(values = c("tanh", "sigmoid", "softplus", "relu", "linear"))
```

## Arguments

- range:

  A two-element vector holding the defaults for the smallest and largest
  possible values.

- trans:

  A `trans` object from the `scales` package, could be `NULL`.

- values:

  A character vector of possible values.

## Value

A `quant_param` or `qual_param` object from the `dials` package.
