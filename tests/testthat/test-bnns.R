test_that("bnns works for regression (linear output with/without normalization(including constant))", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
  new_data <- data.frame(x1 = runif(5), x2 = runif(5))

  result <- bnns(y ~ -1 + x1 + x2, data = df, L = 1, nodes = 2, act_fn = 5, out_act_fn = 1, iter = 1e1, warmup = 5, chains = 1, normalize = FALSE)
  # result_2 <- bnns(y ~ x1 + x2, data = df, L = 4, nodes = rep(2, 4), act_fn = 1:4, out_act_fn = 1, iter = 1e1, warmup = 5, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 1)

  result_summ <- summary.bnns(result)

  expect_true(any(grepl("rmse", names(result_summ$Performance))))
  expect_type(result_summ, "list")
  expect_equal(result_summ$`Number of observations`, nrow(df))

  pred_train <- predict.bnns(result)

  # Test that predictions have correct dimensions
  expect_equal(dim(pred_train)[1], 10) # 10 rows

  # Test that predictions are numeric
  expect_type(pred_train, "double")

  pred_test <- predict.bnns(result, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(pred_test)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(pred_test, "double")

  # # Check class of result
  # expect_s3_class(result_2, "bnns")
  # # Check elements of the returned object
  # expect_true("fit" %in% names(result_2))
  # expect_true("data" %in% names(result_2))
  # expect_equal(result_2$data$out_act_fn, 1)
  # expect_true(result_2$normalize)
  # expect_equal(unname(result_2$x_sd[1]), 1)
})

test_that("bnns works for regression (with custom prior_weights, prior_bias and prior_sigma)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))

  result <- bnns(y ~ -1 + x1 + x2,
    data = df, L = 1, nodes = 2, act_fn = 5, out_act_fn = 1,
    iter = 1e1, warmup = 5, chains = 1,
    prior_weights = list(dist = "uniform", params = list(alpha = -1, beta = 1)),
    prior_bias = list(dist = "cauchy", params = list(mu = 0, sigma = 2.5)),
    prior_sigma = list(dist = "inv_gamma", params = list(alpha = 1, beta = 1))
  )

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 1)
})


test_that("bnns throws appropriate error for missing formula or data", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))

  expect_error(
    bnns(y ~ -1 + x1 + x2, out_act_fn = 1, iter = 1e1, warmup = 5, chains = 1),
    "Both 'formula' and 'data' must be provided."
  )

  expect_error(
    bnns(data = df, out_act_fn = 1, iter = 1e1, warmup = 5, chains = 1),
    "Both 'formula' and 'data' must be provided."
  )
})


test_that("bnns works for binary classification (sigmoid output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = sample(0:1, 10, replace = TRUE))
  new_data <- data.frame(x1 = runif(5), x2 = runif(5))

  result <- bnns(y ~ -1 + x1 + x2, data = df, L = 1, nodes = 2, act_fn = 5, out_act_fn = 2, iter = 1e1, warmup = 5, chains = 1)
  # result_2 <- bnns(y ~ -1 + x1 + x2, data = df, L = 4, nodes = rep(2, 4), act_fn = 1:4, out_act_fn = 2, iter = 1e1, warmup = 5, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 2)
  expect_equal(unique(result$data$y), c(0, 1))

  result_summ <- summary.bnns(result)

  expect_true(any(grepl("accuracy", names(result_summ$Performance))))
  expect_type(result_summ, "list")
  expect_equal(result_summ$`Number of observations`, nrow(df))

  pred_train <- predict.bnns(result)

  # Test that predictions have correct dimensions
  expect_equal(dim(pred_train)[1], 10) # 10 rows

  # Test that predictions are numeric
  expect_type(pred_train, "double")

  pred_test <- predict.bnns(result, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(pred_test)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(pred_test, "double")

  # # Check class of result
  # expect_s3_class(result_2, "bnns")
  # # Check elements of the returned object
  # expect_true("fit" %in% names(result_2))
  # expect_true("data" %in% names(result_2))
  # expect_equal(result_2$data$out_act_fn, 2)
  # expect_equal(unique(result_2$data$y), c(0, 1))
})

test_that("bnns works for multiclass classification (softmax output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = factor(sample(letters[1:3], 10, replace = TRUE)))
  new_data <- data.frame(x1 = runif(5), x2 = runif(5))

  result <- bnns(y ~ -1 + x1 + x2, data = df, L = 1, nodes = 2, act_fn = 5, out_act_fn = 3, iter = 1e1, warmup = 5, chains = 1)
  # result_2 <- bnns(y ~ -1 + x1 + x2, data = df, L = 4, nodes = rep(2, 4), act_fn = 1:4, out_act_fn = 3, iter = 1e1, warmup = 5, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 3)

  result_summ <- summary.bnns(result)

  expect_true(any(grepl("AUC", names(result_summ$Performance))))
  expect_type(result_summ, "list")
  expect_equal(result_summ$`Number of observations`, nrow(df))

  pred_train <- predict.bnns(result)

  # Test that predictions have correct dimensions
  expect_equal(dim(pred_train)[1], 10) # 10 rows

  # Test that predictions are numeric
  expect_type(pred_train, "double")

  pred_test <- predict.bnns(result, newdata = new_data)

  # Test that predictions have correct dimensions
  expect_equal(dim(pred_test)[1], 5) # 5 rows

  # Test that predictions are numeric
  expect_type(pred_test, "double")

  # # Check class of result
  # expect_s3_class(result_2, "bnns")
  # # Check elements of the returned object
  # expect_true("fit" %in% names(result_2))
  # expect_true("data" %in% names(result_2))
  # expect_equal(result_2$data$out_act_fn, 3)
})

test_that("bnns throws errors for incorrect inputs", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = c(0, 1, 2, 1, 0, 0, 1, 2, 1, 0))
  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 2, iter = 1e1, warmup = 5, chains = 1),
    "train_y must have only 0/1 values"
  )

  # Test for multiclass classification with non-factor labels
  df$y <- sample(1:3, 10, replace = TRUE)
  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3),
    "train_y must be a factor"
  )

  df$y <- factor(sample(1:2, 10, replace = TRUE))
  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3),
    "train_y must have at least 3 levels"
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, prior_weights = list(dist = "lognormal", params = list(lower = -1, upper = 1))),
    "Unsupported distribution for weights: lognormal . Supported distributions are: normal, uniform, cauchy"
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, prior_bias = list(dist = "lognormal", params = list(lower = -1, upper = 1))),
    "Unsupported distribution for biases: lognormal . Supported distributions are: normal, uniform, cauchy"
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, L = 0.5),
    "L must be a positive integer"
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, L = 1, nodes = c(4, 2)),
    "nodes must be of length L"
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, L = 1, nodes = 1.5),
    "nodes must be positive integer\\(s\\)"
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, L = 1, act_fn = c(2, 2)),
    "act_fn must be of length L"
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, L = 1, act_fn = 0),
    "act_fn must be a sequence of 1/2/3/4/5"
  )

  df$y[2] <- NA

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, L = 1, act_fn = 1),
    "'data' contains missing values. Please handle them before proceeding."
  )

  df$y <- rnorm(10)
  df$y[2] <- Inf

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 1, L = 1, act_fn = 1),
    "'data' contains invalid values \\(NaN/Inf\\)."
  )

  expect_error(
    bnns(y ~ -1 + x1 + x2, data = lapply(df, function(x)x), out_act_fn = 1, L = 1, act_fn = 1),
    "'data' must be a data.frame."
  )
})
