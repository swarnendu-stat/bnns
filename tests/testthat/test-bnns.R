test_that("bnns works for regression (linear output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))

  result <- bnns(y ~ -1 + x1 + x2, data = df, L = 1, nodes = 2, out_act_fn = 1, iter = 1e2, warmup = 5e1, chains = 1)
  result_2 <- bnns(y ~ -1 + x1 + x2, data = df, L = 4, nodes = rep(2, 4), act_fn = 1:4, out_act_fn = 1, iter = 1e2, warmup = 5e1, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 1)

  # Check class of result
  expect_s3_class(result_2, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result_2))
  expect_true("data" %in% names(result_2))
  expect_equal(result_2$data$out_act_fn, 1)
})

test_that("bnns works for regression (with custom prior_weights and prior_sigma)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))

  result <- bnns(y ~ -1 + x1 + x2,
    data = df, L = 1, nodes = 2, out_act_fn = 1,
    iter = 1e2, warmup = 5e1, chains = 1,
    prior_weights = list(dist = "uniform", params = list(alpha = -1, beta = 1)),
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
    bnns(y ~ -1 + x1 + x2, out_act_fn = 1, iter = 1e2, warmup = 5e1, chains = 1),
    "Both 'formula' and 'data' must be provided."
  )

  expect_error(
    bnns(data = df, out_act_fn = 1, iter = 1e2, warmup = 5e1, chains = 1),
    "Both 'formula' and 'data' must be provided."
  )
})


test_that("bnns works for binary classification (sigmoid output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = sample(0:1, 10, replace = TRUE))

  result <- bnns(y ~ -1 + x1 + x2, data = df, L = 1, nodes = 2, out_act_fn = 2, iter = 1e2, warmup = 5e1, chains = 1)
  result_2 <- bnns(y ~ -1 + x1 + x2, data = df, L = 4, nodes = rep(2, 4), act_fn = 1:4, out_act_fn = 2, iter = 1e2, warmup = 5e1, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 2)
  expect_equal(unique(result$data$y), c(0, 1))

  # Check class of result
  expect_s3_class(result_2, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result_2))
  expect_true("data" %in% names(result_2))
  expect_equal(result_2$data$out_act_fn, 2)
  expect_equal(unique(result_2$data$y), c(0, 1))
})

test_that("bnns works for multiclass classification (softmax output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = factor(sample(letters[1:3], 10, replace = TRUE)))

  result <- bnns(y ~ -1 + x1 + x2, data = df, L = 1, nodes = 2, out_act_fn = 3, iter = 1e2, warmup = 5e1, chains = 1)
  result_2 <- bnns(y ~ -1 + x1 + x2, data = df, L = 4, nodes = rep(2, 4), act_fn = 1:4, out_act_fn = 3, iter = 1e2, warmup = 5e1, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 3)

  # Check class of result
  expect_s3_class(result_2, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result_2))
  expect_true("data" %in% names(result_2))
  expect_equal(result_2$data$out_act_fn, 3)
})

test_that("bnns.default throws errors for incorrect inputs", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = c(0, 1, 2, 1, 0, 0, 1, 2, 1, 0))
  expect_error(
    bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 2, iter = 1e2, warmup = 5e1, chains = 1),
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
    "Unsupported distribution: lognormal . Supported distributions are: normal, uniform, cauchy"
  )
})
