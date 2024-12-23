test_that("summary.bnns works for regression (linear output)", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- rnorm(10)

  result <- summary.bnns(bnns(train_x, train_y, out_act_fn = 1, iter = 2e2, warmup = 1e2, chains = 1))

  expect_true(any(grepl("rmse", names(result$Performance))))
  expect_type(result, "list")
  expect_equal(result$`Number of observations`, length(train_y))
  expect_equal(result$`Performance`$rmse, 0.779606651)
})

test_that("summary.bnns works for binary classification (sigmoid output)", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- sample(0:1, 10, replace = TRUE)

  result <- summary.bnns(bnns(train_x, train_y, out_act_fn = 2, iter = 2e2, warmup = 1e2, chains = 1))

  expect_true(any(grepl("accuracy", names(result$Performance))))
  expect_type(result, "list")
  expect_equal(result$`Number of observations`, length(train_y))
  expect_equal(result$`Performance`$accuracy, 0.8)
})

test_that("summary.bnns works for multiclass classification (softmax output)", {
  set.seed(123)
  train_x <- matrix(rnorm(20), nrow = 10, ncol = 2)
  train_y <- factor(sample(letters[1:3], 10, replace = TRUE))

  result <- summary.bnns(bnns(train_x, train_y, out_act_fn = 3, iter = 2e2, warmup = 1e2, chains = 1))

  expect_true(any(grepl("AUC", names(result$Performance))))
  expect_type(result, "list")
  expect_equal(result$`Number of observations`, length(train_y))
  expect_equal(result$`Performance`$AUC, 0.760416667)
})
