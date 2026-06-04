test_that("generate_stan_code specific wrappers check node length and values", {
  # Test node length mismatch
  expect_error(bnns:::generate_stan_code_cont(2, c(10)))
  expect_error(bnns:::generate_stan_code_bin(2, c(10)))
  expect_error(bnns:::generate_stan_code_cat(2, c(10)))
  
  # Test non-positive nodes
  expect_error(bnns:::generate_stan_code_cont(1, c(0)))
  expect_error(bnns:::generate_stan_code_bin(1, c(0)))
  expect_error(bnns:::generate_stan_code_cat(1, c(0)))
})