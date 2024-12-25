test_that("bnns works for regression (linear output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))

  result <- bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1)

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

  expect_error(bnns(y ~ -1 + x1 + x2, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1),
               "Both 'formula' and 'data' must be provided.")

  expect_error(bnns(data = df, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1),
               "Both 'formula' and 'data' must be provided.")
})


test_that("bnns works for binary classification (sigmoid output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = sample(0:1, 10, replace = TRUE))

  result <- bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 2, iter = 2e2, warmup = 1e2, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 2)
  expect_equal(unique(result$data$y), c(0, 1))
})

test_that("bnns works for multiclass classification (softmax output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = factor(sample(letters[1:3], 10, replace = TRUE)))

  result <- bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, iter = 2e2, warmup = 1e2, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 3)
})

test_that("bnns.default throws errors for incorrect inputs", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = c(0, 1, 2, 1, 0, 0, 1, 2, 1, 0))
  expect_error(bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 2, iter = 2e2, warmup = 1e2, chains = 1),
               "train_y must have only 0/1 values")

  # Test for multiclass classification with non-factor labels
  df$y <- sample(1:3, 10, replace = TRUE)
  expect_error(bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3),
               "train_y must be a factor")
})
