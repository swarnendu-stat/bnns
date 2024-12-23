test_that("tanh works", {
  x <- matrix(1:4, nrow = 2)
  y <- (exp(x) - exp(-x))/(exp(x) + exp(-x))
  expect_equal(tanh(matrix(1:4, nrow = 2)), y)
})

test_that("sigmoid works", {
  x <- matrix(1:4, nrow = 2)
  y <- 1/(1 + exp(-x))
  expect_equal(sigmoid(matrix(1:4, nrow = 2)), y)
})

test_that("softplus works", {
  x <- matrix(1:4, nrow = 2)
  y <- log(1 + exp(x))
  expect_equal(softplus(matrix(1:4, nrow = 2)), y)
})

test_that("relu works for vector", {
  x <- 1:4
  y <- pmax(0, x)
  expect_equal(relu(1:4), y)
})

test_that("relu works for matrix", {
  x <- matrix(1:4, nrow = 2)
  y <- matrix(pmax(0, x), nrow = nrow(x), ncol = ncol(x))
  expect_equal(relu(matrix(1:4, nrow = 2)), y)
})

test_that("softmax works", {
  x <- array(1:64, dim = rep(4, 3))
  y <- x
  for(i in 1:dim(x)[1]){
    for(j in 1:dim(x)[2]){
      y[i, j, ] <- exp(x[i, j, ])/sum(exp(x[i, j, ]))
    }
  }
  expect_equal(softmax_3d(x), y)
})
