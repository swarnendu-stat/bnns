library(testthat)
library(bnns)

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

test_that("generate_stan_code works for various out_act_fn and multiple layers", {
  # out_act_fn = 1, regression, 2 layers
  code1 <- bnns:::generate_stan_code(num_layers = 2, nodes = c(2, 3), out_act_fn = 1)
  expect_true(grepl("vector\\[n\\] y; // Output vector for regression", code1))
  expect_true(grepl("real<lower=0> sigma;", code1))
  expect_true(grepl("normal_lpdf", code1))
  
  # out_act_fn = 2, binary, 1 layer
  code2 <- bnns:::generate_stan_code(num_layers = 1, nodes = c(2), out_act_fn = 2)
  expect_true(grepl("array\\[n\\] int y; // Output vector for binary", code2))
  expect_true(grepl("bernoulli_logit_lpmf", code2))
  
  # out_act_fn = 3, multiclass, 2 layers
  code3 <- bnns:::generate_stan_code(num_layers = 2, nodes = c(2, 3), out_act_fn = 3)
  expect_true(grepl("array\\[n\\] int y; // Output vector for multiclass", code3))
  expect_true(grepl("categorical_logit_lpmf", code3))
  
  # Test prior_weights_dist = horseshoe with all 3 output activation functions
  code_hs1 <- bnns:::generate_stan_code(num_layers = 2, nodes = c(2, 3), out_act_fn = 1, prior_weights_dist = "horseshoe")
  expect_true(grepl("w1_raw .* lambda_w1", code_hs1))
  expect_true(grepl("vector\\[nodes\\[2\\]\\] w_out = w_out_raw .* lambda_w_out \\* tau_w_out", code_hs1))
  
  code_hs2 <- bnns:::generate_stan_code(num_layers = 1, nodes = c(2), out_act_fn = 2, prior_weights_dist = "horseshoe")
  expect_true(grepl("vector\\[nodes\\[1\\]\\] w_out = w_out_raw .* lambda_w_out \\* tau_w_out", code_hs2))
  
  code_hs3 <- bnns:::generate_stan_code(num_layers = 1, nodes = c(2), out_act_fn = 3, prior_weights_dist = "horseshoe")
  expect_true(grepl("to_matrix\\(w_out_raw .* lambda_w_out \\* tau_w_out", code_hs3))
})
