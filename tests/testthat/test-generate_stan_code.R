test_that("code generation works", {
  expect_equal(class(generate_stan_code(1, 16)), "character")
})

test_that("code generation works", {
  expect_equal(class(generate_stan_code(2, c(4, 2))), "character")
})

test_that("shows error for <1 nodes", {
  expect_error(generate_stan_code(2, c(4, 0)))
})

test_that("shows error for nodes matching layers", {
  expect_error(generate_stan_code(2, c(4, 2, 1)))
})

