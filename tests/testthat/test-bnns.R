test_that("bnns.default works for regression (linear output)", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- rnorm(10)

  result <- bnns.default(train_x, train_y, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 1)
})

test_that("bnns works for both formula and default interfaces", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- rnorm(10)
  train_data <- cbind.data.frame(train_y, train_x)

  result_nf <- bnns(train_x, train_y, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1)
  result_f <- bnns(train_y~-1+., data = train_data, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1)

  # Check class of result
  expect_s3_class(result_nf, "bnns")
  expect_s3_class(result_f, "bnns")
})

test_that("bnns throws appropriate error for missing or misaligned inputs", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- rnorm(10)
  train_data <- cbind.data.frame(train_y, train_x)

  # Check class of result
  expect_error(bnns(train_x = train_x, data = train_data, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1),
               "Provide either (formula, data) or (train_x, train_y).", fixed = TRUE)
  expect_error(bnns(formula = y~-1+., train_x = train_x, data = train_data, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1),
               "Provide either (formula, data) or (train_x, train_y), but not both.", fixed = TRUE)
  expect_error(bnns(formula = y~-1+., train_x = train_x, train_y = train_y, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1),
               "Provide either (train_x, train_y) or (formula, data), but not both.", fixed = TRUE)
})


test_that("bnns.default works for binary classification (sigmoid output)", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- sample(0:1, 10, replace = TRUE)

  result <- bnns.default(train_x, train_y, out_act_fn = 2, iter = 2e2, warmup = 1e2, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 2)
  expect_equal(unique(result$data$y), c(0, 1))
})

test_that("bnns.default works for multiclass classification (softmax output)", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- factor(sample(letters[1:3], 10, replace = TRUE))

  result <- bnns.default(train_x, train_y, out_act_fn = 3, iter = 2e2, warmup = 1e2, chains = 1)

  # Check class of result
  expect_s3_class(result, "bnns")
  # Check elements of the returned object
  expect_true("fit" %in% names(result))
  expect_true("data" %in% names(result))
  expect_equal(result$data$out_act_fn, 3)
  expect_true(is.factor(train_y))
})

test_that("bnns.default throws errors for incorrect inputs", {
  set.seed(123)
  train_x <- matrix(rnorm(100), nrow = 20, ncol = 5)

  # Test for binary classification with invalid labels
  train_y <- c(0, 1, 2, 1, 0)
  expect_error(bnns.default(train_x, train_y, out_act_fn = 2, iter = 2e2, warmup = 1e2, chains = 1),
               "train_y must have only 0/1 values")

  # Test for multiclass classification with non-factor labels
  train_y <- sample(1:3, 20, replace = TRUE)
  expect_error(bnns.default(train_x, train_y, out_act_fn = 3),
               "train_y must be a factor")
})
