test_that("bnns stops on invalid inputs and edge cases", {
  df <- data.frame(x = rnorm(10), y = rnorm(10))
  
  # Missing parameters
  expect_error(bnns(y ~ x))
  expect_error(bnns(data = df))
  
  # Data contains missing values or infinite values
  df_na <- df
  df_na$x[1] <- NA
  expect_error(bnns(y ~ x, data = df_na), "contains missing values")
  
  df_inf <- df
  df_inf$x[1] <- Inf
  expect_error(bnns(y ~ x, data = df_inf), "invalid values")
  
  # Invalid data format
  expect_error(bnns(y ~ x, data = as.matrix(df)), "must be a data.frame")
  
  # Invalid prior validation
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "unknown", params = list(mean = 0, sd = 1))))
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "normal", params = list(mean = 0))))
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "uniform", params = list(alpha = 0))))
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "cauchy", params = list(mu = 0))))
  
  expect_error(bnns(y ~ x, data = df, prior_bias = list(dist = "unknown", params = list(mean = 0, sd = 1))))
  expect_error(bnns(y ~ x, data = df, prior_sigma = list(dist = "half_normal", params = list(mean = 0))))
  expect_error(bnns(y ~ x, data = df, prior_sigma = list(dist = "inv_gamma", params = list(alpha = 1))))
})

test_that("plot.bnns errors when appropriate", {
  # Mock model for checking classification vs regression specific behavior
  mock_fit <- list(
    data = list(out_act_fn = 1), # 1 = regression
    levels = NULL
  )
  class(mock_fit) <- "bnns"
  
  expect_error(plot(mock_fit, type = "pred_prob"), "only applicable for classification models")
})

test_that("predict.bnns validates type parameter", {
  mock_fit <- list(
    data = list(out_act_fn = 1),
    levels = NULL
  )
  class(mock_fit) <- "bnns"
  
  expect_error(predict(mock_fit, type = "prob"), "only applicable for classification models")
  expect_error(predict(mock_fit, type = "class"), "only applicable for classification models")
})