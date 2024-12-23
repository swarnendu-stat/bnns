# general
test_that("generates code for continuous response correctly", {
  stan_code <- generate_stan_code(num_layers = 2, nodes = c(10, 5), out_act_fn = 1)

  # Tests
  expect_type(stan_code, "character")  # Stan code should be a character string
  expect_match(stan_code, "data \\{", fixed = FALSE)  # Check data block exists
  expect_match(stan_code, "y ~ normal", fixed = FALSE)  # Check binary response model
  expect_match(stan_code, "matrix\\[n, nodes\\[1\\]\\] z1", fixed = FALSE)  # Check intermediate layers
})

test_that("generates code for binary response correctly", {
  stan_code <- generate_stan_code(num_layers = 2, nodes = c(10, 5), out_act_fn = 2)

  # Tests
  expect_type(stan_code, "character")  # Stan code should be a character string
  expect_match(stan_code, "data \\{", fixed = FALSE)  # Check data block exists
  expect_match(stan_code, "y ~ bernoulli_logit", fixed = FALSE)  # Check binary response model
  expect_match(stan_code, "matrix\\[n, nodes\\[1\\]\\] z1", fixed = FALSE)  # Check intermediate layers
})

test_that("generates code for categorical response correctly", {
  stan_code <- generate_stan_code(num_layers = 2, nodes = c(10, 5), out_act_fn = 3)

  # Tests
  expect_type(stan_code, "character")  # Stan code should be a character string
  expect_match(stan_code, "data \\{", fixed = FALSE)  # Check data block exists
  expect_no_match(stan_code, "y ~ categorical_logit", fixed = FALSE)  # Check binary response model
  expect_match(stan_code, "y\\[i\\] ~ categorical_logit", fixed = FALSE)  # Check binary response model
  expect_match(stan_code, "matrix\\[n, nodes\\[1\\]\\] z1", fixed = FALSE)  # Check intermediate layers
})


# continuous
test_that("code generation works", {
  expect_equal(class(generate_stan_code_cont(1, 16)), "character")
})

test_that("code generation works", {
  expect_equal(class(generate_stan_code_cont(2, c(4, 2))), "character")
})

test_that("shows error for <1 nodes", {
  expect_error(generate_stan_code_cont(2, c(4, 0)))
})

test_that("shows error for nodes matching layers", {
  expect_error(generate_stan_code_cont(2, c(4, 2, 1)))
})

# binary
test_that("code generation works", {
  expect_equal(class(generate_stan_code_bin(1, 16)), "character")
})

test_that("code generation works", {
  expect_equal(class(generate_stan_code_bin(2, c(4, 2))), "character")
})

test_that("shows error for <1 nodes", {
  expect_error(generate_stan_code_bin(2, c(4, 0)))
})

test_that("shows error for nodes matching layers", {
  expect_error(generate_stan_code_bin(2, c(4, 2, 1)))
})

# categorical
test_that("code generation works", {
  expect_equal(class(generate_stan_code_cat(1, 16)), "character")
})

test_that("code generation works", {
  expect_equal(class(generate_stan_code_cat(2, c(4, 2))), "character")
})

test_that("shows error for <1 nodes", {
  expect_error(generate_stan_code_cat(2, c(4, 0)))
})

test_that("shows error for nodes matching layers", {
  expect_error(generate_stan_code_cat(2, c(4, 2, 1)))
})
