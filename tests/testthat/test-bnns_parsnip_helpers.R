library(testthat)
library(bnns)

test_that("translate_activation works correctly", {
  expect_equal(bnns:::translate_activation("tanh"), 1L)
  expect_equal(bnns:::translate_activation(c("sigmoid", "relu")), c(2L, 4L))
  expect_error(bnns:::translate_activation("unknown"), "Unknown activation")
  expect_equal(bnns:::translate_activation(1:2), 1:2)
})

test_that("translate_out_activation works correctly", {
  expect_equal(bnns:::translate_out_activation("linear"), 1L)
  expect_equal(bnns:::translate_out_activation(c("sigmoid", "softmax")), c(2L, 3L))
  expect_error(bnns:::translate_out_activation("unknown"), "Unknown out_act_fn")
  expect_equal(bnns:::translate_out_activation(1L), 1L)
})

test_that("detect_output_activation works correctly", {
  expect_equal(bnns:::detect_output_activation(rnorm(10)), 1L)
  expect_equal(bnns:::detect_output_activation(factor(c("A", "B"))), 2L)
  expect_equal(bnns:::detect_output_activation(c("A", "B")), 2L)
  expect_equal(bnns:::detect_output_activation(factor(c("A", "B", "C"))), 3L)
  expect_equal(bnns:::detect_output_activation(c("A", "B", "C")), 3L)
})
