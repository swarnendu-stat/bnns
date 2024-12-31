test_that("works for continuous case", {
  obs <- c(3.2, 4.1, 5.6)
  pred <- c(3.0, 4.3, 5.5)
  measure <- measure_cont(obs, pred)
  expect_equal(measure$rmse, 0.173205081)
  expect_equal(measure$mae, 0.166666667)
})

test_that("works for binary case", {
  obs <- c(1, 0, 1, 1, 0)
  pred <- c(0.9, 0.4, 0.8, 0.7, 0.3)
  cut <- 0.5
  measure <- measure_bin(obs, pred, cut)
  expect_equal(measure$accuracy, 1)
  expect_equal(measure$AUC, 1)
})

test_that("works for categorical case", {
  obs <- factor(c("A", "B", "C"), levels = LETTERS[1:3])
  pred <- matrix(
    c(
      0.8, 0.1, 0.1,
      0.2, 0.6, 0.2,
      0.7, 0.2, 0.1
    ),
    nrow = 3, byrow = TRUE
  )
  measure <- measure_cat(obs, pred)
  expect_equal(measure$log_loss, 1.01218476)
  expect_equal(measure$AUC, 0.75)
})
