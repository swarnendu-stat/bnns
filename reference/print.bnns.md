# Print Method for `"bnns"` Objects

Displays a summary of a fitted Bayesian Neural Network (BNN) model,
including the function call and the Stan fit details.

## Usage

``` r
# S3 method for class 'bnns'
print(x, ...)
```

## Arguments

- x:

  An object of class `"bnns"`, typically the result of a call to
  [`bnns.default`](https://swarnendu-stat.github.io/bnns/reference/bnns.default.md).

- ...:

  Additional arguments (currently not used).

## Value

The function is called for its side effects and does not return a value.
It prints the following:

- The function call used to generate the `"bnns"` object.

- A summary of the Stan fit object stored in `x$fit`.

## See also

[`bnns`](https://swarnendu-stat.github.io/bnns/reference/bnns.md),
[`summary.bnns`](https://swarnendu-stat.github.io/bnns/reference/summary.bnns.md)

## Examples

``` r
# \donttest{
# Example usage:
data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
model <- bnns(y ~ -1 + x1 + x2,
  data = data, L = 1, nodes = 2, act_fn = 2,
  iter = 1e1, warmup = 5, chains = 1
)
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
#> Chain 1: 
#> Chain 1: Gradient evaluation took 1.5e-05 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: WARNING: No variance estimation is
#> Chain 1:          performed for num_warmup < 20
#> Chain 1: 
#> Chain 1: Iteration: 1 / 10 [ 10%]  (Warmup)
#> Chain 1: Iteration: 2 / 10 [ 20%]  (Warmup)
#> Chain 1: Iteration: 3 / 10 [ 30%]  (Warmup)
#> Chain 1: Iteration: 4 / 10 [ 40%]  (Warmup)
#> Chain 1: Iteration: 5 / 10 [ 50%]  (Warmup)
#> Chain 1: Iteration: 6 / 10 [ 60%]  (Sampling)
#> Chain 1: Iteration: 7 / 10 [ 70%]  (Sampling)
#> Chain 1: Iteration: 8 / 10 [ 80%]  (Sampling)
#> Chain 1: Iteration: 9 / 10 [ 90%]  (Sampling)
#> Chain 1: Iteration: 10 / 10 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 0 seconds (Warm-up)
#> Chain 1:                0 seconds (Sampling)
#> Chain 1:                0 seconds (Total)
#> Chain 1: 
print(model)
#> Call:
#> bnns.default(formula = y ~ -1 + x1 + x2, data = data, L = 1, 
#>     nodes = 2, act_fn = 2, iter = 10, warmup = 5, chains = 1)
#> 
#> Stan fit:
#> Inference for Stan model: anon_model.
#> 1 chains, each with iter=10; warmup=5; thin=1; 
#> post-warmup draws per chain=5, total post-warmup draws=5.
#> 
#>              mean se_mean   sd   2.5%    25%    50%   75% 97.5% n_eff Rhat
#> w1[1,1]      0.49    0.17 0.29   0.19   0.19   0.53  0.78  0.78     3  Inf
#> w1[1,2]      0.26    0.08 0.14   0.17   0.17   0.23  0.23  0.49     3  Inf
#> w1[2,1]      0.42    0.24 0.44   0.09   0.11   0.11  0.91  0.91     3  Inf
#> w1[2,2]      1.27    0.33 0.58   0.65   0.65   1.51  1.77  1.77     3  Inf
#> b1[1]       -1.17    0.23 0.42  -1.66  -1.35  -1.35 -0.74 -0.74     3  Inf
#> b1[2]       -0.09    0.30 0.51  -0.57  -0.57  -0.20  0.45  0.45     3  Inf
#> w_out[1]    -0.84    0.08 0.13  -0.97  -0.97  -0.83 -0.72 -0.72     3  Inf
#> w_out[2]    -0.04    0.15 0.28  -0.26  -0.24  -0.24  0.26  0.26     3  Inf
#> b_out[1]     0.76    0.20 0.38   0.48   0.48   0.72  0.72  1.33     3  Inf
#> log_lik[1]  -1.72    0.22 0.41  -2.37  -1.57  -1.57 -1.50 -1.50     3  Inf
#> log_lik[2]  -1.41    0.10 0.18  -1.51  -1.51  -1.48 -1.48 -1.13     3  Inf
#> log_lik[3]  -1.26    0.08 0.14  -1.40  -1.40  -1.22 -1.22 -1.08     3  Inf
#> log_lik[4]  -1.23    0.08 0.14  -1.41  -1.26  -1.26 -1.09 -1.09     3  Inf
#> log_lik[5]  -2.62    0.22 0.41  -2.85  -2.85  -2.76 -2.76 -1.98     3  Inf
#> log_lik[6]  -1.67    0.19 0.35  -2.21  -1.51  -1.51 -1.51 -1.51     3  Inf
#> log_lik[7]  -1.18    0.05 0.09  -1.25  -1.25  -1.24 -1.09 -1.09     3  Inf
#> log_lik[8]  -1.59    0.17 0.32  -2.09  -1.49  -1.49 -1.41 -1.41     3  Inf
#> log_lik[9]  -1.47    0.11 0.21  -1.58  -1.58  -1.56 -1.56 -1.14     3  Inf
#> log_lik[10] -1.19    0.05 0.09  -1.29  -1.29  -1.12 -1.11 -1.11     3  Inf
#> y_rep[1]     0.67    1.28 2.39  -2.05  -1.75   1.88  2.52  2.76     3 0.71
#> y_rep[2]     0.65    0.21 0.40   0.13   0.49   0.64  0.93  1.10     3 0.98
#> y_rep[3]     0.50    0.54 1.00  -0.62   0.10   0.36  0.73  1.90     3 0.94
#> y_rep[4]    -1.41    1.00 1.86  -3.81  -2.55  -1.08 -0.16  0.61     3 2.33
#> y_rep[5]     0.11    0.61 1.14  -1.41  -0.67   0.70  1.01  1.02     3 0.77
#> y_rep[6]    -0.83    1.10 2.06  -2.34  -2.16  -1.76 -0.48  2.31     3 0.92
#> y_rep[7]     0.57    0.54 1.00  -0.51  -0.30   0.87  0.87  1.84     3 1.22
#> y_rep[8]     0.72    0.82 1.54  -0.59  -0.57   0.14  1.74  2.77     3 0.89
#> y_rep[9]    -0.06    0.78 1.45  -1.96  -0.77   0.06  0.83  1.58     3 0.90
#> y_rep[10]    0.44    0.56 1.04  -1.06   0.28   0.62  0.91  1.53     3 0.94
#> sigma        1.25    0.06 0.12   1.16   1.17   1.17  1.38  1.38     3  Inf
#> lp__        -9.64    0.69 1.28 -11.28 -10.00 -10.00 -8.38 -8.38     3  Inf
#> 
#> Samples were drawn using NUTS(diag_e) at Sat Jun  6 07:32:54 2026.
#> For each parameter, n_eff is a crude measure of effective sample size,
#> and Rhat is the potential scale reduction factor on split chains (at 
#> convergence, Rhat=1).
# }
```
