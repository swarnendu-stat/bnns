# Generic Function for Fitting Bayesian Neural Network Models

This is a generic function for fitting Bayesian Neural Network (BNN)
models. It dispatches to methods based on the class of the input data.

## Usage

``` r
bnns(
  formula,
  data,
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

- formula:

  A symbolic description of the model to be fitted. The formula should
  specify the response variable and predictors (e.g., `y ~ x1 + x2`).
  `y` must be continuous for regression (`out_act_fn = 1`), numeric 0/1
  for binary classification (`out_act_fn = 2`), and factor with at least
  3 levels for multi-classification (`out_act_fn = 3`).

- data:

  A data frame containing the variables in the model.

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

  An integer specifying the activation function for the output layer.
  Options are:

  - `1` for linear (default)

  - `2` for sigmoid

  - `3` for softmax

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
    Supported values are `"normal"`, `"uniform"`, `"cauchy"`, and
    `"horseshoe"`.

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
    Supported values are `"normal"`, `"uniform"`, `"cauchy"`, and
    `"horseshoe"`.

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

The result of the method dispatched by the class of the input data.
Typically, this would be an object of class `"bnns"` containing the
fitted model and associated information.

## Details

The function serves as a generic interface to different methods of
fitting Bayesian Neural Networks. The specific method dispatched depends
on the class of the input arguments, allowing for flexibility in the
types of inputs supported.

## References

1.  Bishop, C.M., 1995. Neural networks for pattern recognition. Oxford
    university press.

2.  Carpenter, B., Gelman, A., Hoffman, M.D., Lee, D., Goodrich, B.,
    Betancourt, M., Brubaker, M.A., Guo, J., Li, P. and Riddell,
    A., 2017. Stan: A probabilistic programming language. Journal of
    statistical software, 76.

3.  Neal, R.M., 2012. Bayesian learning for neural networks (Vol. 118).
    Springer Science & Business Media.

## See also

[`bnns.default`](https://swarnendu-stat.github.io/bnns/reference/bnns.default.md)

## Examples

``` r
# \donttest{
# Example usage with formula interface:
data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
model <- bnns(y ~ -1 + x1 + x2,
  data = data, L = 1, nodes = 2, act_fn = 1,
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
# }
# See the documentation for bnns.default for more details on the default implementation.
```
