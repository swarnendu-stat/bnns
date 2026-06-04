# tests/testthat/test-parsnip_integration.R
library(testthat)
library(parsnip)
library(bnns)
library(dplyr)

test_that("bnns regression parsnip integration works", {
  # Fast sampling parameters for testing
  reg_spec <- mlp(mode = "regression", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  fit_reg <- fit(reg_spec, mpg ~ hp + wt, data = mtcars)
  
  # Check if model fitted properly
  expect_s3_class(fit_reg$fit, "bnns")
  
  # Test predictions
  preds <- predict(fit_reg, new_data = mtcars[1:5, ])
  
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 5)
  expect_true(is.numeric(preds$.pred))
})

test_that("bnns binary classification parsnip integration works", {
  # Create a binary target
  df_bin <- iris %>% 
    filter(Species != "virginica") %>%
    mutate(Species = droplevels(Species))
  
  bin_spec <- mlp(mode = "classification", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  fit_bin <- fit(bin_spec, Species ~ Sepal.Length + Sepal.Width, data = df_bin)
  
  # 1. Test Hard Class Predictions
  class_preds <- predict(fit_bin, new_data = df_bin[1:5, ], type = "class")
  
  expect_s3_class(class_preds, "tbl_df")
  expect_equal(names(class_preds), ".pred_class")
  expect_equal(nrow(class_preds), 5)
  expect_true(is.factor(class_preds$.pred_class))
  expect_equal(levels(class_preds$.pred_class), levels(df_bin$Species))
  
  # 2. Test Probabilistic Predictions
  prob_preds <- predict(fit_bin, new_data = df_bin[1:5, ], type = "prob")
  
  expect_s3_class(prob_preds, "tbl_df")
  expect_equal(names(prob_preds), c(".pred_setosa", ".pred_versicolor"))
  expect_equal(nrow(prob_preds), 5)
  
  # Verify probabilities sum to 1
  row_sums <- rowSums(prob_preds)
  expect_true(all(abs(row_sums - 1) < 1e-6))
})

test_that("bnns multiclass classification parsnip integration works", {
  multi_spec <- mlp(mode = "classification", hidden_units = 2, epochs = 20) %>%
    set_engine("bnns", warmup = 10, refresh = 0, chains = 1)
  
  fit_multi <- fit(multi_spec, Species ~ Sepal.Length + Sepal.Width, data = iris)
  
  # 1. Test Hard Class Predictions
  class_preds <- predict(fit_multi, new_data = iris[1:5, ], type = "class")
  
  expect_s3_class(class_preds, "tbl_df")
  expect_equal(names(class_preds), ".pred_class")
  expect_equal(nrow(class_preds), 5)
  expect_true(is.factor(class_preds$.pred_class))
  expect_equal(levels(class_preds$.pred_class), levels(iris$Species))
  
  # 2. Test Probabilistic Predictions
  prob_preds <- predict(fit_multi, new_data = iris[1:5, ], type = "prob")
  
  expect_s3_class(prob_preds, "tbl_df")
  expect_equal(names(prob_preds), c(".pred_setosa", ".pred_versicolor", ".pred_virginica"))
  expect_equal(nrow(prob_preds), 5)
  
  # Verify probabilities sum to 1
  row_sums <- rowSums(prob_preds)
  expect_true(all(abs(row_sums - 1) < 1e-6))
})