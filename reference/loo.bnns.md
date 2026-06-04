# Leave-One-Out Cross-Validation (LOO) for bnns models

Leave-One-Out Cross-Validation (LOO) for bnns models

## Usage

``` r
# S3 method for class 'bnns'
loo(x, ...)
```

## Arguments

- x:

  A fitted `bnns` model object.

- ...:

  Additional arguments passed to
  [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html).

## Value

A `loo` object containing model comparison metrics.
