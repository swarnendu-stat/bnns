
<!-- README.md is generated from README.Rmd. Please edit that file -->

# bnns <a href="https://swarnendu-stat.github.io/bnns/"><img src="man/figures/logo.png" align="right" height="139" alt="bnns website" /></a>

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/swarnendu-stat/bnns/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/swarnendu-stat/bnns/actions/workflows/R-CMD-check.yaml)
[![Codecov test
coverage](https://codecov.io/gh/swarnendu-stat/bnns/graph/badge.svg)](https://app.codecov.io/gh/swarnendu-stat/bnns)
<!-- badges: end -->

The `bnns` package provides tools to fit Bayesian Neural Networks (BNNs)
for regression and classification problems. It is designed to be
flexible, supporting various network architectures, activation
functions, and output types, making it suitable for both simple and
complex data analysis tasks.

## Features

- Support for multi-layer neural networks with customizable
  architecture.
- Choice of activation functions (e.g., sigmoid, ReLU, tanh).
- Outputs for regression (continuous response) and classification
  (binary and multiclass).
- Bayesian inference, providing posterior distributions for predictions
  and parameters.
- Applications in domains such as clinical trials, predictive modeling,
  and more.

## Installation (stable CRAN version)

To install the `bnns` package from CRAN, use the following:

``` r
# install.packages("bnns")
```

## Installation (development version)

To install the `bnns` package from GitHub, use the following:

``` r
# Install devtools if not already installed
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

# Install bnns
devtools::install_github("swarnendu-stat/bnns")
#> Using GitHub PAT from the git credential store.
#> Downloading GitHub repo swarnendu-stat/bnns@HEAD
#> 
#> ── R CMD build ─────────────────────────────────────────────────────────────────
#> * checking for file ‘/tmp/Rtmp72ouVc/remotes423e532d8878/swarnendu-stat-bnns-a75ca26/DESCRIPTION’ ... OK
#> * preparing ‘bnns’:
#> * checking DESCRIPTION meta-information ... OK
#> * checking for LF line-endings in source and make files and shell scripts
#> * checking for empty or unneeded directories
#> * building ‘bnns_0.0.0.9000.tar.gz’
#> Installing package into '/tmp/RtmpzIYroh/temp_libpath11573c8d4159'
#> (as 'lib' is unspecified)
```

## Getting Started

### 1. Simulate Data

Below is an example of how to simulate data for regression:

``` r
set.seed(123)
df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
```

### 2. Fit a BNN Model

To fit a Bayesian Neural Network:

``` r
library(bnns)

model <- bnns(y ~ -1 + x1 + x2,
  data = df, L = 2, nodes = c(10, 8), act_fn = c(2, 3), out_act_fn = 1,
  iter = 1e2, warmup = 5e1, chains = 1, seed = 123
)
```

### 3. Model Summary

Summarize the fitted model:

``` r
summary(model)
#> Call:
#> bnns.default(formula = y ~ -1 + x1 + x2, data = df, L = 2, nodes = c(10, 
#>     8), act_fn = c(2, 3), out_act_fn = 1, iter = 100, warmup = 50, 
#>     chains = 1, seed = 123)
#> 
#> Data Summary:
#> Number of observations: 10 
#> Number of features: 2 
#> 
#> Network Architecture:
#> Number of hidden layers: 2 
#> Nodes per layer: 10, 8 
#> Activation functions: 2, 3 
#> Output activation function: 1 
#> 
#> Posterior Summary (Key Parameters):
#>                 mean    se_mean        sd       2.5%        25%         50%
#> w_out[1] -0.02324795 0.13129860 1.0762213 -1.6600031 -1.0103546 -0.18735995
#> w_out[2]  0.07270977 0.14981448 0.8350675 -1.3137496 -0.4830793 -0.06217806
#> w_out[3]  0.11283857 0.11428048 0.7623686 -1.0485410 -0.5333280  0.08261137
#> w_out[4]  0.20054171 0.11334737 0.7588289 -1.2580135 -0.2220071  0.16581899
#> w_out[5] -0.06208152 0.16932869 1.0527341 -1.8664903 -0.9103519  0.06347047
#> w_out[6]  0.30302264 0.15389760 1.0756999 -1.6702569 -0.4103843  0.19989870
#> w_out[7]  0.30786594 0.11187634 0.8229034 -1.0878378 -0.3385778  0.32939318
#> w_out[8] -0.15144312 0.12836466 0.9245842 -1.7927356 -0.6969301 -0.20455107
#> b_out     0.24797547 0.13604812 0.9077106 -1.5935206 -0.4190687  0.23167117
#> sigma     0.95245930 0.04625639 0.3037826  0.5781807  0.7346287  0.92544357
#>                75%    97.5%    n_eff      Rhat
#> w_out[1] 0.7441097 1.840549 67.18664 0.9865655
#> w_out[2] 0.7052703 1.624580 31.06960 0.9923414
#> w_out[3] 0.7150443 1.618166 44.50265 1.0045883
#> w_out[4] 0.7419605 1.455819 44.81927 0.9798358
#> w_out[5] 0.6306202 1.720851 38.65238 0.9824005
#> w_out[6] 1.0225610 2.465771 48.85607 1.0054149
#> w_out[7] 0.9952039 1.614277 54.10299 0.9812693
#> w_out[8] 0.3860326 1.242786 51.88024 0.9814778
#> b_out    0.9793195 1.889829 44.51535 0.9851646
#> sigma    1.0812277 1.639066 43.13027 0.9828852
#> 
#> Model Fit Information:
#> Iterations: 100 
#> Warmup: 50 
#> Thinning: 1 
#> Chains: 1 
#> 
#> Predictive Performance:
#> RMSE (training): 0.7956568 
#> MAE (training): 0.6356295 
#> 
#> Notes:
#> Check convergence diagnostics for parameters with high R-hat values.
```

### 4. Predictions

Make predictions using the trained model:

``` r
pred <- predict(model)
```

### 5. Visualization

Visualize true vs predicted values for regression:

``` r
plot(df$y, rowMeans(pred), main = "True vs Predicted", xlab = "True Values", ylab = "Predicted Values")
abline(0, 1, col = "red")
```

<img src="man/figures/README-unnamed-chunk-8-1.png" width="100%" />

## Applications

### Regression Example (with custom priors)

Use `bnns` for regression analysis to model continuous outcomes, such as
predicting patient biomarkers in clinical trials.

``` r
model <- bnns(y ~ -1 + x1 + x2,
  data = df, L = 2, nodes = c(10, 8), act_fn = c(2, 3), out_act_fn = 1,
  iter = 1e2, warmup = 5e1, chains = 1, seed = 123,
  prior_weights = list(dist = "uniform", params = list(alpha = -1, beta = 1)),
  prior_sigma = list(dist = "inv_gamma", params = list(alpha = 1, beta = 1))
)
```

### Classification Example

For binary or multiclass classification, set the `out_act_fn` to `2`
(binary) or `3` (multiclass). For example:

``` r
# Simulate binary classification data
df <- data.frame(x1 = runif(10), x2 = runif(10), y = sample(0:1, 10, replace = TRUE))

# Fit a binary classification BNN
model <- bnns(y ~ -1 + x1 + x2, data = df, L = 2, nodes = c(16, 8), act_fn = c(3, 2), out_act_fn = 2, iter = 1e2, warmup = 5e1, chains = 1)
```

### Clinical Trial Applications

Explore posterior probabilities to estimate treatment effects or success
probabilities in clinical trials. For example, calculate the posterior
probability of achieving a clinically meaningful outcome in a given
population.

## Documentation

- Detailed vignettes are available to guide through various applications
  of the package.
- See `help(bnns)` for more information about the `bnns` function and
  its arguments.

## Contributing

Contributions are welcome! Please raise issues or submit pull requests
on [GitHub](https://github.com/swarnendu-stat/bnns).

## License

This package is licensed under the Apache License. See `LICENSE` for
details.
