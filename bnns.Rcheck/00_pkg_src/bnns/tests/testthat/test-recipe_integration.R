# tests/testthat/test-recipe_integration.R
library(testthat)
library(parsnip)
library(bnns)
library(dplyr)

test_that("bnns regression works with recipes and step_normalize", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("workflows")
  skip_on_cran()
  
  library(recipes)
  library(workflows)
  
  # 1. Define a recipe with normalization
  rec <- recipe(mpg ~ hp + wt + disp, data = mtcars) %>%
    step_normalize(all_numeric_predictors())
  
  # 2. Define the model
  reg_spec <- mlp(mode = "regression", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  # 3. Create and fit the workflow
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(reg_spec)
  
  fit_wf <- fit(wf, data = mtcars)
  
  # Check if model engine fitted properly
  engine_fit <- extract_fit_engine(fit_wf)
  expect_s3_class(engine_fit, "bnns")
  
  # 4. Predict on new data
  preds <- predict(fit_wf, new_data = mtcars[1:5, ])
  
  # Assertions
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 5)
  expect_true(is.numeric(preds$.pred))
})

test_that("bnns classification works with recipes and step_normalize", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("workflows")
  skip_on_cran()

  library(recipes)
  library(workflows)

  df_bin <- iris %>% 
    filter(Species != "virginica") %>%
    mutate(Species = droplevels(Species))
  
  rec <- recipe(Species ~ Sepal.Length + Sepal.Width, data = df_bin) %>%
    step_normalize(all_numeric_predictors())
  
  class_spec <- mlp(mode = "classification", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(class_spec)
  
  fit_wf <- fit(wf, data = df_bin)
  
  # Predict probabilities to ensure factor levels passed through the recipe unchanged
  preds <- predict(fit_wf, new_data = df_bin[1:5, ], type = "prob")
  
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), c(".pred_setosa", ".pred_versicolor"))
  expect_equal(nrow(preds), 5)
})

test_that("bnns regression handles missing data with step_impute_mean", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("workflows")
  skip_on_cran()

  library(recipes)
  library(workflows)

  # Create data with NAs
  mtcars_na <- mtcars
  mtcars_na$hp[c(1, 5, 10)] <- NA
  mtcars_na$wt[c(2, 6, 12)] <- NA

  # 1. Define a recipe with mean imputation
  rec <- recipe(mpg ~ hp + wt + disp, data = mtcars_na) %>%
    step_impute_mean(all_numeric_predictors())
  
  # 2. Define the model
  reg_spec <- mlp(mode = "regression", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  # 3. Create and fit the workflow
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(reg_spec)
  
  fit_wf <- fit(wf, data = mtcars_na)
  
  # Check if model engine fitted properly
  engine_fit <- extract_fit_engine(fit_wf)
  expect_s3_class(engine_fit, "bnns")
  
  # 4. Predict on new data containing NAs
  preds <- predict(fit_wf, new_data = mtcars_na[1:5, ])
  
  # Assertions
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 5)
  expect_true(is.numeric(preds$.pred))
  expect_false(any(is.na(preds$.pred)))
})

test_that("bnns classification handles missing factor levels with step_unknown", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("workflows")
  skip_on_cran()

  library(recipes)
  library(workflows)

  # Create binary classification data with a categorical predictor containing NAs
  df_bin <- iris %>% 
    filter(Species != "virginica") %>%
    mutate(
      Species = droplevels(Species),
      Category = factor(rep(c("A", "B", NA, "A"), 25))
    )
  
  # 1. Define a recipe with unknown imputation for categorical predictors
  rec <- recipe(Species ~ Sepal.Length + Sepal.Width + Category, data = df_bin) %>%
    step_unknown(all_nominal_predictors())
  
  # 2. Define the model
  class_spec <- mlp(mode = "classification", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  # 3. Create and fit the workflow
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(class_spec)
  
  fit_wf <- fit(wf, data = df_bin)
  
  # Check if model engine fitted properly
  engine_fit <- extract_fit_engine(fit_wf)
  expect_s3_class(engine_fit, "bnns")
  
  # 4. Predict hard classes on new data containing NAs
  preds <- predict(fit_wf, new_data = df_bin[1:5, ], type = "class")
  
  # Assertions to ensure predictions executed successfully and without NAs
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred_class")
  expect_equal(nrow(preds), 5)
  expect_false(any(is.na(preds$.pred_class)))
})