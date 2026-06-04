# Using bnns with tidymodels

## Introduction

The `bnns` package provides a fully integrated `parsnip` engine,
allowing you to fit Bayesian Neural Networks (BNNs) seamlessly within
the tidymodels ecosystem.

By registering the `"bnns"` engine for the `mlp()` (Multi-Layer
Perceptron) model specification, you can leverage the powerful data
preprocessing capabilities of `recipes`, manage modeling pipelines with
`workflows`, and evaluate performance using `yardstick`, all while
benefiting from the robust probabilistic inference provided by `bnns`
and Stan.

### Setup

To get started, load the `tidymodels` meta-package along with `bnns`.
Loading `bnns` automatically registers the engine with `parsnip`.

``` r

library(tidymodels)
library(bnns)
```

## Regression

Let’s start with a regression task using the built-in `mtcars` dataset
to predict miles per gallon (`mpg`).

### 1. Specify the Model

We use the `mlp()` function to define a neural network. We map the
standard `parsnip` arguments to `bnns` parameters: - `hidden_units` maps
to `nodes`. - `epochs` maps to `iter` (total Stan iterations). -
`activation` maps to `act_fn` (e.g., `"relu"`, `"tanh"`, `"sigmoid"`).

Additional Stan-specific arguments, like `chains`, `warmup`, and
`cores`, can be passed directly to `set_engine()`.

``` r

bnn_reg_spec <- mlp(
  mode = "regression",
  hidden_units = 5,
  epochs = 500,
  activation = "relu"
) %>% 
  set_engine(
    engine = "bnns", 
    chains = 2, 
    warmup = 250, 
    refresh = 0,
    seed = 123
  )

bnn_reg_spec
```

### 2. Create a Workflow and Fit

We can combine our model specification with a simple formula into a
`workflow()`.

*(Note: The following chunk is not evaluated by default to keep the
package build times within CRAN limits, but you can run it locally!)*

``` r

bnn_reg_wf <- workflow() %>%
  add_model(bnn_reg_spec) %>%
  add_formula(mpg ~ hp + wt + cyl + disp)

# Fit the model
bnn_reg_fit <- fit(bnn_reg_wf, data = mtcars)

bnn_reg_fit
```

### 3. Predict

Predictions follow the standard `tidymodels` format, returning a tibble
with a `.pred` column containing the posterior mean prediction.

``` r

predictions <- predict(bnn_reg_fit, new_data = mtcars)
head(predictions)
```

## Classification

The `"bnns"` engine also supports binary and multiclass classification.
We’ll demonstrate a multiclass example using the `iris` dataset to
predict the `Species`.

### 1. Specify the Model

For classification, simply change the mode to `"classification"`. Under
the hood, `bnns` will automatically detect that the outcome is a factor
and adjust the output activation function to `softmax` (or `sigmoid` for
binary data).

``` r

bnn_class_spec <- mlp(
  mode = "classification",
  hidden_units = 4,
  epochs = 500,
  activation = "tanh"
) %>% 
  set_engine(
    engine = "bnns", 
    chains = 1, 
    warmup = 200, 
    refresh = 0,
    seed = 456
  )
```

### 2. Create a Workflow and Fit

We can easily pair this with a `recipe` for data preprocessing. For
example, neural networks benefit significantly from centered and scaled
predictors. While `bnns` standardizes internally by default, explicitly
handling it in a recipe is standard practice in `tidymodels`.

``` r

iris_rec <- recipe(Species ~ ., data = iris) %>%
  step_normalize(all_numeric_predictors())

bnn_class_wf <- workflow() %>%
  add_model(bnn_class_spec) %>%
  add_recipe(iris_rec)

bnn_class_fit <- fit(bnn_class_wf, data = iris)
```

### 3. Predict Classes and Probabilities

With classification models, you can generate both hard class predictions
and soft class probabilities.

``` r

# 1. Predict hard classes (returns a .pred_class factor column)
class_preds <- predict(bnn_class_fit, new_data = iris, type = "class")
head(class_preds)

# 2. Predict class probabilities (returns .pred_{Level} columns)
prob_preds <- predict(bnn_class_fit, new_data = iris, type = "prob")
head(prob_preds)
```

You can then bind these predictions to the original dataset to evaluate
metrics like accuracy or ROC-AUC using the `yardstick` package.

``` r

eval_data <- bind_cols(iris, class_preds, prob_preds)

accuracy(eval_data, truth = Species, estimate = .pred_class)
roc_auc(eval_data, truth = Species, .pred_setosa:.pred_virginica)
```

Enjoy seamless Bayesian Neural Network modeling integrated directly into
your favorite data science workflows!
