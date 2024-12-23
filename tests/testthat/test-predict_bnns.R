# Running NN

data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
model_f <- bnns(y ~ -1 + x1 + x2, data = data, L = 1, nodes = 2, act_fn = 2, iter = 2e2, warmup = 1e2, chains = 1)
model_nf <- bnns(with(data, cbind(x1, x2)), data$y, L = 1, nodes = 2, act_fn = 2, iter = 2e2, warmup = 1e2, chains = 1)
model_nf_4 <- bnns(with(data, cbind(x1, x2)), data$y, L = 4, nodes = rep(2, 4), act_fn = 1:4, iter = 2e2, warmup = 1e2, chains = 1)
new_data <- data.frame(x1 = runif(5), x2 = runif(5))

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

test_that("predict.bnns handles formula and non-formula cases", {
  # Formula-based newdata
  formula_predictions <- capture_predictions(model_f, newdata = new_data)

  # Non-formula newdata
  matrix_predictions <- capture_predictions(model_nf, newdata = as.matrix(new_data))

  # Test that both produce consistent results
  expect_equal(rowMeans(formula_predictions), rowMeans(matrix_predictions))
})

test_that("predict.bnns works with for more than 1 layer with different activation functions", {
  predictions <- capture_predictions(model_nf_4, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(predictions)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(predictions, "double")
})
