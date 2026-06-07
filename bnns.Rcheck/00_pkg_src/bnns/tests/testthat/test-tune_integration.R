# tests/testthat/test-tune_integration.R
library(testthat)
library(parsnip)
library(bnns)
library(dplyr)

test_that("bnns regression cross-validation works with tune_grid", {
  skip_if_not_installed("rsample")
  skip_if_not_installed("tune")
  skip_if_not_installed("workflows")
  skip_on_cran()
  
  library(rsample)
  library(tune)
  library(workflows)
  
  # Fast sampling parameters for testing. Tuning the hidden_units parameter.
  reg_spec <- mlp(mode = "regression", hidden_units = tune(), epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  reg_wf <- workflow() %>%
    add_model(reg_spec) %>%
    add_formula(mpg ~ hp + wt)
  
  set.seed(123)
  cv_folds <- vfold_cv(mtcars, v = 2)
  
  reg_grid <- data.frame(hidden_units = c(2, 3))
  
  tune_res <- tune_grid(
    reg_wf,
    resamples = cv_folds,
    grid = reg_grid,
    control = control_grid(save_pred = TRUE)
  )
  
  expect_s3_class(tune_res, "tune_results")
  
  metrics <- collect_metrics(tune_res)
  expect_true(nrow(metrics) > 0)
  
  preds <- collect_predictions(tune_res)
  expect_true(nrow(preds) > 0)
  expect_true(".pred" %in% names(preds))
})

test_that("bnns classification cross-validation works with fit_resamples", {
  skip_if_not_installed("rsample")
  skip_if_not_installed("tune")
  skip_if_not_installed("workflows")
  skip_on_cran()

  library(rsample)
  library(tune)
  library(workflows)

  # Create a binary target
  df_bin <- iris %>% 
    filter(Species != "virginica") %>%
    mutate(Species = droplevels(Species))
  
  class_spec <- mlp(mode = "classification", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  class_wf <- workflow() %>%
    add_model(class_spec) %>%
    add_formula(Species ~ Sepal.Length + Sepal.Width)
  
  set.seed(123)
  cv_folds <- vfold_cv(df_bin, v = 2)
  
  res <- fit_resamples(
    class_wf,
    resamples = cv_folds,
    control = control_resamples(save_pred = TRUE)
  )
  
  expect_s3_class(res, "tune_results")
  
  metrics <- collect_metrics(res)
  expect_true(nrow(metrics) > 0)
  
  preds <- collect_predictions(res)
  expect_true(nrow(preds) > 0)
  expect_true(all(c(".pred_class", "id", ".row") %in% names(preds)))
})