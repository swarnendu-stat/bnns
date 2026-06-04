# Plot diagnostics for a fitted Bayesian Neural Network

Generates Markov Chain Monte Carlo (MCMC) trace plots, posterior density
plots, Posterior Predictive Checks (PPC), or predicted probability
distributions for the fitted model.

## Usage

``` r
# S3 method for class 'bnns'
plot(
  x,
  type = c("trace", "density", "posterior_predictive", "pred_prob"),
  pars = NULL,
  ...
)
```

## Arguments

- x:

  A fitted `bnns` model object.

- type:

  Character string indicating the type of plot. Options are `"trace"`
  for MCMC trace plots, `"density"` for posterior density plots,
  `"posterior_predictive"` for Posterior Predictive Checks, and
  `"pred_prob"` for visualizing the predicted class probability
  distributions (classification only).

- pars:

  A character vector of parameter names to include in the plot. By
  default, this focuses on the output layer (`"w_out"`, `"b_out"`, and
  `"sigma"`) to avoid cluttering the plot device with hundreds of hidden
  layer weights.

- ...:

  Additional arguments passed to
  [`stan_trace`](https://mc-stan.org/rstan/reference/stan_plot.html),
  [`stan_dens`](https://mc-stan.org/rstan/reference/stan_plot.html), or
  [`ppc_dens_overlay`](https://mc-stan.org/bayesplot/reference/PPC-distributions.html).

## Value

A `ggplot` object containing the requested diagnostic plots.
