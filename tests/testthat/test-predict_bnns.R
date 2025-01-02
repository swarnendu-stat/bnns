# Running NN

data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
model_f <- bnns(y ~ -1 + x1 + x2, data = data, L = 1, nodes = 2, act_fn = 2, iter = 1e2, warmup = 5e1, chains = 1, normalize = FALSE)
model_f_2 <- bnns(y ~ -1 + x1 + x2, data = data, L = 4, nodes = rep(2, 4), act_fn = 1:4, iter = 1e2, warmup = 5e1, chains = 1)
new_data <- data.frame(x1 = runif(5), x2 = runif(5))

data_bin <- data.frame(x1 = runif(10), x2 = runif(10), y = sample(0:1, 10, replace = TRUE))
model_bin <- bnns(y ~ -1 + x1 + x2, data = data_bin, L = 4, nodes = rep(2, 4), act_fn = 1:4, iter = 1e2, warmup = 5e1, chains = 1, out_act_fn = 2)

data_cat <- data.frame(x1 = runif(10), x2 = runif(10), y = factor(sample(LETTERS[1:3], 10, replace = TRUE)))
model_cat <- bnns(y ~ -1 + x1 + x2, data = data_cat, L = 4, nodes = rep(2, 4), act_fn = 1:4, iter = 1e2, warmup = 5e1, chains = 1, out_act_fn = 3)

# Redirect output for testing predictions
capture_predictions <- function(object, newdata = NULL) {
  predict.bnns(object, newdata = newdata)
}

# Test script for predict.bnns
test_that("predict.bnns produces predictions for training data", {
  predictions <- capture_predictions(model_f)

  # Test that predictions have correct dimensions
  expect_equal(dim(predictions)[1], 10) # 10 rows

  # Test that predictions are numeric
  expect_type(predictions, "double")
})

test_that("predict.bnns works with newdata", {
  predictions <- capture_predictions(model_f, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(predictions)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(predictions, "double")
})

test_that("predict.bnns works with for more than 1 layer with different activation functions", {
  predictions <- capture_predictions(model_f_2, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(predictions)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(predictions, "double")
})

test_that("predict.bnns works with for more than 1 layer with different activation functions for the categorical case", {
  predictions <- capture_predictions(model_cat, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(predictions)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(predictions, "double")
})

test_that("predict.bnns works with for more than 1 layer with different activation functions for the binary case", {
  predictions <- capture_predictions(model_bin, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(predictions)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(predictions, "double")
})
