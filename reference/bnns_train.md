# Internal function for training the BNN

This function performs the actual fitting of the Bayesian Neural
Network. It is called by the exported bnns methods and is not intended
for direct use.

## Usage

``` r
bnns_train(
  train_x,
  train_y,
  L = 1,
  nodes = rep(2, L),
  act_fn = rep(2, L),
  out_act_fn = 1,
  algorithm = c("NUTS", "HMC"),
  iter = 1000,
  warmup = 200,
  thin = 1,
  chains = 2,
  cores = 2,
  seed = 123,
  prior_weights = NULL,
  prior_bias = NULL,
  prior_sigma = NULL,
  verbose = FALSE,
  refresh = max(iter/10, 1),
  normalize = TRUE,
  backend = c("rstan", "cmdstanr"),
  use_gpu = FALSE,
  opencl_ids = c(0, 0),
  ...
)
```

## Arguments

- train_x:

  A numeric matrix representing the input features (predictors) for
  training. Rows correspond to observations, and columns correspond to
  features.

- train_y:

  A numeric vector representing the target values for training. Its
  length must match the number of rows in `train_x`.

- L:

  An integer specifying the number of hidden layers in the neural
  network. Default is 1.

- nodes:

  An integer or vector specifying the number of nodes in each hidden
  layer. If a single value is provided, it is applied to all layers.
  Default is 16.

- act_fn:

  An integer or vector specifying the activation function(s) for the
  hidden layers. Options are:

  - `1` for tanh

  - `2` for sigmoid (default)

  - `3` for softplus

  - `4` for ReLU

  - `5` for linear

- out_act_fn:

  An integer or character string specifying the activation function for
  the output layer. Options are:

  - `1` or `"linear"` for linear (default)

  - `2` or `"sigmoid"` for sigmoid

  - `3` or `"softmax"` for softmax

- algorithm:

  A character string specifying the MCMC algorithm. Options are `"NUTS"`
  (default) or `"HMC"`.

- iter:

  An integer specifying the total number of iterations for the Stan
  sampler. Default is `1e3`.

- warmup:

  An integer specifying the number of warmup iterations for the Stan
  sampler. Default is `2e2`.

- thin:

  An integer specifying the thinning interval for Stan samples. Default
  is 1.

- chains:

  An integer specifying the number of Markov chains. Default is 2.

- cores:

  An integer specifying the number of CPU cores to use for parallel
  sampling. Default is 2.

- seed:

  An integer specifying the random seed for reproducibility. Default is
  123.

- prior_weights:

  A list specifying the prior distribution for the weights in the neural
  network. The list must include two components:

  - `dist`: A character string specifying the distribution type.
    Supported values are `"normal"`, `"uniform"`, and `"cauchy"`.

  - `params`: A named list specifying the parameters for the chosen
    distribution:

    - For `"normal"`: Provide `mean` (mean of the distribution) and `sd`
      (standard deviation).

    - For `"uniform"`: Provide `alpha` (lower bound) and `beta` (upper
      bound).

    - For `"cauchy"`: Provide `mu` (location parameter) and `sigma`
      (scale parameter).

  For the `"horseshoe"` prior, `params` is not needed as it uses a
  standard half-Cauchy setup. If `prior_weights` is `NULL`, the default
  prior is a `normal(0, 1)` distribution. For example:

  - `list(dist = "normal", params = list(mean = 0, sd = 1))`

  - `list(dist = "uniform", params = list(alpha = -1, beta = 1))`

  - `list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))`

- prior_bias:

  A list specifying the prior distribution for the biases in the neural
  network. The list must include two components:

  - `dist`: A character string specifying the distribution type.
    Supported values are `"normal"`, `"uniform"`, and `"cauchy"`.

  - `params`: A named list specifying the parameters for the chosen
    distribution:

    - For `"normal"`: Provide `mean` (mean of the distribution) and `sd`
      (standard deviation).

    - For `"uniform"`: Provide `alpha` (lower bound) and `beta` (upper
      bound).

    - For `"cauchy"`: Provide `mu` (location parameter) and `sigma`
      (scale parameter).

  If `prior_bias` is `NULL`, the default prior is a `normal(0, 1)`
  distribution. For example:

  - `list(dist = "normal", params = list(mean = 0, sd = 1))`

  - `list(dist = "uniform", params = list(alpha = -1, beta = 1))`

  - `list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))`

- prior_sigma:

  A list specifying the prior distribution for the `sigma` parameter in
  regression models (`out_act_fn = 1`). This allows for setting priors
  on the standard deviation of the residuals. The list must include two
  components:

  - `dist`: A character string specifying the distribution type.
    Supported values are `"half-normal"` and `"inverse-gamma"`.

  - `params`: A named list specifying the parameters for the chosen
    distribution:

    - For `"half-normal"`: Provide `sd` (standard deviation of the
      half-normal distribution).

    - For `"inverse-gamma"`: Provide `shape` (shape parameter) and
      `scale` (scale parameter).

  If `prior_sigma` is `NULL`, the default prior is a `half-normal(0, 1)`
  distribution. For example:

  - `list(dist = "half_normal", params = list(mean = 0, sd = 1))`

  - `list(dist = "inv_gamma", params = list(alpha = 1, beta = 1))`

- verbose:

  TRUE or FALSE: flag indicating whether to print intermediate output
  from Stan on the console, which might be helpful for model debugging.

- refresh:

  refresh (integer) can be used to control how often the progress of the
  sampling is reported (i.e. show the progress every refresh
  iterations). By default, refresh = max(iter/10, 1). The progress
  indicator is turned off if refresh \<= 0.

- normalize:

  Logical. If `TRUE` (default), the input predictors are normalized to
  have zero mean and unit variance before training. Normalization
  ensures stable and efficient Bayesian sampling by standardizing the
  input scale, which is particularly beneficial for neural network
  training. If `FALSE`, no normalization is applied, and it is assumed
  that the input data is already pre-processed appropriately.

- backend:

  A character string specifying the Stan backend to use. Options are
  `"rstan"` (default) or `"cmdstanr"`.

- use_gpu:

  Logical. If `TRUE`, enables OpenCL for GPU acceleration. Default is
  `FALSE`. (Requires the `"cmdstanr"` backend).

- opencl_ids:

  A vector of two integers specifying the OpenCL platform and device
  IDs. Default is `c(0, 0)`.

- ...:

  Currently not in use.

## Value

An object of class `"bnns"` containing the following components:

- `fit`:

  The fitted Stan model object.

- `call`:

  The matched call.

- `data`:

  A list containing the Stan data used in the model.

## Details

The function uses the `generate_stan_code` function to dynamically
generate Stan code based on the specified number of layers and nodes.
Stan is then used to fit the Bayesian Neural Network.

## See also

[`stan`](https://mc-stan.org/rstan/reference/stan.html)

## Examples

``` r
# \donttest{
# Example usage:
train_x <- matrix(runif(20), nrow = 10, ncol = 2)
train_y <- rnorm(10)
model <- bnns::bnns_train(train_x, train_y,
  L = 1, nodes = 2, act_fn = 2,
  iter = 1e1, warmup = 5, chains = 1
)
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
#> Chain 1: 
#> Chain 1: Gradient evaluation took 2.3e-05 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
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

# Access Stan model fit
model$fit
#> Inference for Stan model: anon_model.
#> 1 chains, each with iter=10; warmup=5; thin=1; 
#> post-warmup draws per chain=5, total post-warmup draws=5.
#> 
#>               mean se_mean   sd   2.5%    25%    50%   75% 97.5% n_eff  Rhat
#> w1[1,1]      -0.04    0.43 0.80  -0.73  -0.73  -0.38  0.82  0.82     3  5.51
#> w1[1,2]       0.29    0.05 0.09   0.20   0.20   0.30  0.30  0.42     3  0.71
#> w1[2,1]      -0.21    0.20 0.38  -0.63  -0.39  -0.39  0.18  0.18     3  0.76
#> w1[2,2]       0.00    0.19 0.35  -0.24  -0.24  -0.06 -0.06  0.54     3  1.31
#> b1[1]         0.20    0.16 0.29  -0.05  -0.05   0.21  0.21  0.62     3  0.73
#> b1[2]         1.04    0.24 0.45   0.63   0.63   0.91  1.51  1.51     3  3.74
#> w_out[1]      0.56    0.32 0.60  -0.03  -0.03   0.72  0.72  1.33     3  2.38
#> w_out[2]     -0.54    0.19 0.36  -1.03  -0.61  -0.61 -0.20 -0.20     3  2.06
#> b_out[1]      0.15    0.33 0.63  -0.40  -0.40  -0.10  0.82  0.82     3  5.06
#> log_lik[1]   -3.39    0.62 1.16  -4.66  -3.93  -3.93 -2.16 -2.16     3  0.76
#> log_lik[2]   -3.61    0.53 0.99  -4.47  -4.47  -4.01 -2.55 -2.55     3  5.20
#> log_lik[3]   -1.26    0.10 0.19  -1.45  -1.45  -1.21 -1.21 -1.01     3  2.38
#> log_lik[4]   -1.14    0.09 0.17  -1.24  -1.24  -1.19 -1.19 -0.88     3  0.88
#> log_lik[5]   -1.48    0.18 0.35  -1.85  -1.85  -1.32 -1.19 -1.19     3  1.29
#> log_lik[6]   -1.33    0.18 0.34  -1.68  -1.68  -1.20 -1.20 -0.94     3  3.13
#> log_lik[7]   -1.35    0.16 0.30  -1.67  -1.67  -1.19 -1.19 -1.05     3  0.79
#> log_lik[8]   -1.21    0.12 0.23  -1.40  -1.40  -1.20 -1.20 -0.88     3  1.65
#> log_lik[9]   -1.67    0.26 0.49  -2.21  -2.21  -1.32 -1.31 -1.31     3 79.57
#> log_lik[10]  -1.48    0.26 0.48  -1.99  -1.99  -1.24 -1.24 -0.95     3  3.99
#> y_rep[1]     -0.73    0.42 0.78  -1.41  -1.40  -1.04  0.05  0.15     3  4.87
#> y_rep[2]      0.42    0.98 1.83  -1.49  -0.16  -0.13  0.65  3.07     3  1.33
#> y_rep[3]     -1.15    0.93 1.73  -3.83  -1.03  -0.51 -0.32  0.20     3  1.04
#> y_rep[4]      0.07    0.46 0.86  -1.15  -0.38   0.49  0.71  0.74     3  1.15
#> y_rep[5]     -0.30    0.78 1.46  -2.02  -0.95  -0.82  1.09  1.28     3 12.06
#> y_rep[6]     -0.53    0.20 0.37  -0.99  -0.64  -0.64 -0.19 -0.15     3  0.71
#> y_rep[7]      1.05    1.00 1.87  -0.55  -0.30   0.67  1.37  3.81     3  1.36
#> y_rep[8]     -0.54    0.52 0.96  -1.59  -1.57  -0.09  0.27  0.29     3  1.27
#> y_rep[9]     -0.44    0.33 0.62  -1.04  -1.00  -0.56  0.17  0.25     3  0.73
#> y_rep[10]     0.63    0.71 1.33  -0.80  -0.31   0.65  1.05  2.43     3  1.47
#> sigma         1.20    0.09 0.16   0.96   1.23   1.23  1.32  1.32     3  0.80
#> lp__        -11.05    0.57 1.07 -12.14 -12.14 -10.99 -9.99 -9.99     3  2.42
#> 
#> Samples were drawn using NUTS(diag_e) at Fri Jun 12 20:00:42 2026.
#> For each parameter, n_eff is a crude measure of effective sample size,
#> and Rhat is the potential scale reduction factor on split chains (at 
#> convergence, Rhat=1).
# }
```
