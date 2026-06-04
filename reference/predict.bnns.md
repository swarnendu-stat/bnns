# Predictions from a fitted Bayesian Neural Network

Predictions from a fitted Bayesian Neural Network

## Usage

``` r
# S3 method for class 'bnns'
predict(
  object,
  newdata = NULL,
  type = c("samples", "mean", "median", "quantile", "prob", "class"),
  quantiles = c(0.025, 0.975),
  ...
)
```

## Arguments

- object:

  A fitted `bnns` model object.

- newdata:

  A data frame containing new data for prediction. If not provided, the
  predictions will be generated using the training data.

- type:

  Character string indicating the type of prediction. Options are
  `"samples"` (default), `"mean"`, `"median"`, `"quantile"`, `"prob"`
  (class probabilities), or `"class"` (predicted class labels).

- quantiles:

  Numeric vector of probabilities used when `type = "quantile"`. Default
  is `c(0.025, 0.975)`.

- ...:

  Additional arguments passed to internal prediction methods.

## Value

- For `type = "samples"`: A matrix (regression/binary) or 3D array
  (multiclass) of posterior predictions.

- For `type = "mean"` or `"median"`: A vector or matrix of aggregated
  predictions. For classification tasks, `type = "mean"` returns the
  posterior mean class probabilities.

- For `type = "quantile"`: A matrix or array of quantiles.

- For `type = "prob"`: A matrix of class probabilities (for
  classification models).

- For `type = "class"`: A vector of predicted class labels (for
  classification models).
