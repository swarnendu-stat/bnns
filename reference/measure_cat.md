# Measure Performance for Multi-Class Classification Models

Evaluates the performance of a multi-class classification model using
log loss and multiclass AUC.

## Usage

``` r
measure_cat(obs, pred)
```

## Arguments

- obs:

  A factor vector of observed class labels. Each level represents a
  unique class.

- pred:

  A numeric matrix of predicted probabilities, where each row
  corresponds to an observation, and each column corresponds to a class.
  The number of columns must match the number of levels in `obs`.

## Value

A list containing:

- `log_loss`:

  The negative log-likelihood averaged across observations.

- `ROC`:

  ROC generated using
  [`pROC::roc`](https://rdrr.io/pkg/pROC/man/roc.html)

- `AUC`:

  The multiclass Area Under the Curve (AUC) as computed by
  [`pROC::multiclass.roc`](https://rdrr.io/pkg/pROC/man/multiclass.html).

## Details

The log loss is calculated as: \$\$-\frac{1}{N} \sum\_{i=1}^N
\sum\_{c=1}^C y\_{ic} \log(p\_{ic})\$\$ where \\y\_{ic}\\ is 1 if
observation \\i\\ belongs to class \\c\\, and \\p\_{ic}\\ is the
predicted probability for that class.

The AUC is computed using the
[`pROC::multiclass.roc`](https://rdrr.io/pkg/pROC/man/multiclass.html)
function, which provides an overall measure of model performance for
multiclass classification.

## Examples

``` r
library(pROC)
#> Type 'citation("pROC")' for a citation.
#> 
#> Attaching package: ‘pROC’
#> The following objects are masked from ‘package:stats’:
#> 
#>     cov, smooth, var
obs <- factor(c("A", "B", "C"), levels = LETTERS[1:3])
pred <- matrix(
  c(
    0.8, 0.1, 0.1,
    0.2, 0.6, 0.2,
    0.7, 0.2, 0.1
  ),
  nrow = 3, byrow = TRUE
)
measure_cat(obs, pred)
#> $log_loss
#> [1] 1.012185
#> 
#> $ROC
#> 
#> Call:
#> multiclass.roc.default(response = obs, predictor = `colnames<-`(data.frame(pred),     levels(obs)))
#> 
#> Data: multivariate predictor `colnames<-`(data.frame(pred), levels(obs)) with 3 levels of obs: A, B, C.
#> Multi-class area under the curve: 0.75
#> 
#> $AUC
#> [1] 0.75
#> 
# Returns: list(log_loss = 1.012185, ROC = <ROC>, AUC = 0.75)
```
