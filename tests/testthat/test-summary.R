test_that("summary.bnns works for regression (linear output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))

  result <- summary.bnns(bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 1, iter = 1e2, warmup = 5e1, chains = 1))

  expect_true(any(grepl("rmse", names(result$Performance))))
  expect_type(result, "list")
  expect_equal(result$`Number of observations`, nrow(df))
})

test_that("summary.bnns works for binary classification (sigmoid output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = sample(0:1, 10, replace = TRUE))

  result <- summary.bnns(bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 2, iter = 1e2, warmup = 5e1, chains = 1))

  expect_true(any(grepl("accuracy", names(result$Performance))))
  expect_type(result, "list")
  expect_equal(result$`Number of observations`, nrow(df))
})

test_that("summary.bnns works for multiclass classification (softmax output)", {
  set.seed(123)
  df <- data.frame(x1 = runif(10), x2 = runif(10), y = factor(sample(letters[1:3], 10, replace = TRUE)))

  result <- summary.bnns(bnns(y ~ -1 + x1 + x2, data = df, out_act_fn = 3, iter = 1e2, warmup = 5e1, chains = 1))

  expect_true(any(grepl("AUC", names(result$Performance))))
  expect_type(result, "list")
  expect_equal(result$`Number of observations`, nrow(df))
})
