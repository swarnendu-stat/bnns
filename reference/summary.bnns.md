# Summary of a Bayesian Neural Network (BNN) Model

Provides a comprehensive summary of a fitted Bayesian Neural Network
(BNN) model, including details about the model call, data, network
architecture, posterior distributions, and model fitting information.

## Usage

``` r
# S3 method for class 'bnns'
summary(object, ...)
```

## Arguments

- object:

  An object of class `bnns`, representing a fitted Bayesian Neural
  Network model.

- ...:

  Additional arguments (currently unused).

## Value

A list (returned invisibly) containing the following elements:

- `"Number of observations"`: The number of observations in the training
  data.

- `"Number of features"`: The number of features in the training data.

- `"Number of hidden layers"`: The number of hidden layers in the neural
  network.

- `"Nodes per layer"`: A comma-separated string representing the number
  of nodes in each hidden layer.

- `"Activation functions"`: A comma-separated string representing the
  activation functions used in each hidden layer.

- `"Output activation function"`: The activation function used in the
  output layer.

- `"Stanfit Summary"`: A summary of the Stan model, including key
  parameter posterior distributions.

- `"Iterations"`: The total number of iterations used for sampling in
  the Bayesian model.

- `"Warmup"`: The number of iterations used as warmup in the Bayesian
  model.

- `"Thinning"`: The thinning interval used in the Bayesian model.

- `"Chains"`: The number of Markov chains used in the Bayesian model.

- `"Performance"`: Predictive performance metrics, which vary based on
  the output activation function.

The function also prints the summary to the console.

## Details

The function prints the following information:

- **Call:** The original function call used to fit the model.

- **Data Summary:** Number of observations and features in the training
  data.

- **Network Architecture:** Structure of the BNN including the number of
  hidden layers, nodes per layer, and activation functions.

- **Posterior Summary:** Summarized posterior distributions of key
  parameters (e.g., weights, biases, and noise parameter).

- **Model Fit Information:** Bayesian sampling details, including the
  number of iterations, warmup period, thinning, and chains.

- **Notes:** Remarks and warnings, such as checks for convergence
  diagnostics.

## See also

[`bnns`](https://swarnendu-stat.github.io/bnns/reference/bnns.md),
[`print.bnns`](https://swarnendu-stat.github.io/bnns/reference/print.bnns.md)

## Examples

``` r
# \donttest{
# Fit a Bayesian Neural Network
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

# Get a summary of the model
summary(model)
#> Call:
#> bnns.default(formula = y ~ -1 + x1 + x2, data = data, L = 1, 
#>     nodes = 2, act_fn = 2, iter = 10, warmup = 5, chains = 1)
#> 
#> Data Summary:
#> Number of observations: 10 
#> Number of features: 2 
#> 
#> Network Architecture:
#> Number of hidden layers: 1 
#> Nodes per layer: 2 
#> Activation functions: 2 
#> Output activation function: 1 
#> 
#> Posterior Summary (Key Parameters):
#>                 mean   se_mean        sd       2.5%        25%         50%
#> w_out[1] -0.10658560 0.1862499 0.3481852 -0.5423493 -0.2655034 -0.17091634
#> w_out[2] -0.02894635 0.1811263 0.3386068 -0.3100852 -0.3100852 -0.15416228
#> b_out[1] -0.09364786 0.2286431 0.4274373 -0.6352952 -0.3674245  0.06477078
#> sigma     1.09904539 0.1607550 0.3005238  0.8602735  0.9974541  0.99745412
#>                 75%     97.5%   n_eff      Rhat
#> w_out[1] 0.23830087 0.2383009 3.49485 1.4373560
#> w_out[2] 0.15901638 0.4394278 3.49485 0.7191104
#> b_out[1] 0.06477078 0.3977091 3.49485 0.8303403
#> sigma    1.03474525 1.5619625 3.49485 0.8098036
#> 
#> Model Fit Information:
#> Iterations: 10 
#> Warmup: 5 
#> Thinning: 1 
#> Chains: 1 
#> 
#> Predictive Performance:
#> RMSE (training): 1.07225 
#> MAE (training): 0.7732455 
#> 
#> Notes:
#> Check convergence diagnostics for parameters with high R-hat values.
# }
```
