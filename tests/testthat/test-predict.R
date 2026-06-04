test_that("predict.bnns works for regression", {
  skip_on_cran()
  
  df <- data.frame(x = runif(20), y = rnorm(20))
  fit <- bnns(y ~ x, data = df, L = 1, nodes = 2, out_act_fn = 1, iter = 20, warmup = 10, chains = 1, refresh = 0)
  
  pred_mean <- predict(fit, type = "mean")
  expect_equal(length(pred_mean), 20)
  
  pred_quant <- predict(fit, type = "quantile")
  expect_equal(dim(pred_quant), c(20, 2))
  
  expect_error(predict(fit, type = "prob"), "only applicable for classification models")
  expect_error(predict(fit, type = "class"), "only applicable for classification models")
})

test_that("predict.bnns works for binary classification", {
  skip_on_cran()
  
  df <- data.frame(x = runif(20), y = factor(sample(c("A", "B"), 20, replace = TRUE)))
  fit <- bnns(y ~ x, data = df, L = 1, nodes = 2, out_act_fn = 2, iter = 20, warmup = 10, chains = 1, refresh = 0)
  
  pred_prob <- predict(fit, type = "prob")
  expect_equal(dim(pred_prob), c(20, 2))
  
  pred_class <- predict(fit, type = "class")
  expect_equal(length(pred_class), 20)
})

test_that("predict.bnns works for multiclass classification", {
  skip_on_cran()
  
  df <- data.frame(x = runif(30), y = factor(sample(c("A", "B", "C"), 30, replace = TRUE)))
  fit <- bnns(y ~ x, data = df, L = 1, nodes = 2, out_act_fn = 3, iter = 20, warmup = 10, chains = 1, refresh = 0)
  
  pred_prob <- predict(fit, type = "prob")
  expect_equal(dim(pred_prob), c(30, 3))
  
  pred_class <- predict(fit, type = "class")
  expect_equal(length(pred_class), 30)
})