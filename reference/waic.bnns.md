# Watanabe-Akaike Information Criterion (WAIC) for bnns models

Watanabe-Akaike Information Criterion (WAIC) for bnns models

## Usage

``` r
# S3 method for class 'bnns'
waic(x, ...)
```

## Arguments

- x:

  A fitted `bnns` model object.

- ...:

  Additional arguments passed to
  [`loo::waic()`](https://mc-stan.org/loo/reference/waic.html).

## Value

A `waic` object containing model comparison metrics.
