# Measure Performance for Binary Classification Models

Evaluates the performance of a binary classification model using a
confusion matrix and accuracy.

## Usage

``` r
measure_bin(obs, pred, cut = 0.5)
```

## Arguments

- obs:

  A numeric or integer vector of observed binary class labels (0 or 1).

- pred:

  A numeric vector of predicted probabilities for the positive class.

- cut:

  A numeric threshold (between 0 and 1) to classify predictions into
  binary labels.

## Value

A list containing:

- `conf_mat`:

  A confusion matrix comparing observed and predicted class labels.

- `accuracy`:

  The proportion of correct predictions.

- `ROC`:

  ROC generated using
  [`pROC::roc`](https://rdrr.io/pkg/pROC/man/roc.html)

- `AUC`:

  Area under the ROC curve.

## Examples

``` r
obs <- c(1, 0, 1, 1, 0)
pred <- c(0.9, 0.4, 0.8, 0.7, 0.3)
cut <- 0.5
measure_bin(obs, pred, cut)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> $conf_mat
#>    pred_label
#> obs 0 1
#>   0 2 0
#>   1 0 3
#> 
#> $accuracy
#> [1] 1
#> 
#> $ROC
#> 
#> Call:
#> roc.default(response = obs, predictor = pred)
#> 
#> Data: pred in 2 controls (obs 0) < 3 cases (obs 1).
#> Area under the curve: 1
#> 
#> $AUC
#> [1] 1
#> 
# Returns: list(conf_mat = <confusion matrix>, accuracy = 1, ROC = <ROC>, AUC = 1)
```
