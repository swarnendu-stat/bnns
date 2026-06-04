test_that("save_load error cases", {
  expect_error(save_bnns(1, "test.rds"), "Object must be of class 'bnns'.")
  saveRDS(1, "test.rds")
  expect_error(load_bnns("test.rds"), "The file does not contain a valid 'bnns' object.")
  unlink("test.rds")
})

test_that("bnns prior and input validation", {
  df <- data.frame(x = 1:10, y = 1:10)
  
  # bnns_train validations
  expect_error(bnns_train(), "Argument train_x is missing")
  expect_error(bnns_train(matrix(1:10)), "Argument train_y is missing")
  expect_error(bnns_train(matrix(1:10), 1:10, L = 1.5), "L must be a positive integer")
  expect_error(bnns_train(matrix(1:10), 1:10, L = 2, nodes = 2), "nodes must be of length L")
  expect_error(bnns_train(matrix(1:10), 1:10, L = 1, nodes = -2), "nodes must be positive integer")
  expect_error(bnns_train(matrix(1:10), 1:10, L = 1, act_fn = c(1,2)), "act_fn must be of length L")
  expect_error(bnns_train(matrix(1:10), 1:10, L = 1, act_fn = 6), "act_fn must be a sequence")
  
  # bnns validations
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "unknown", params = list())), "Unsupported distribution for weights")
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "normal")), "'prior_weights' must be a list with elements 'dist' and 'params'.")
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "normal", params = list(mean = 0))), "For 'normal' distribution, 'params' must contain 'mean' and 'sd'.")
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "uniform", params = list(mean = 0))), "For 'uniform' distribution, 'params' must contain 'alpha' and 'beta'.")
  expect_error(bnns(y ~ x, data = df, prior_weights = list(dist = "cauchy", params = list(mean = 0))), "For 'cauchy' distribution, 'params' must contain 'mu' and 'sigma'.")

  expect_error(bnns(y ~ x, data = df, prior_bias = list(dist = "unknown", params = list())), "Unsupported distribution for biases")
  
  expect_error(bnns(y ~ x, data = df, prior_sigma = list(dist = "half_normal", params = list(mean = 0))), "For 'half_normal' distribution, 'params' must contain 'mean' and 'sd'.")
  expect_error(bnns(y ~ x, data = df, prior_sigma = list(dist = "inv_gamma", params = list(mean = 0))), "For 'inv_gamma' distribution, 'params' must contain 'alpha' and 'beta'.")
  expect_error(bnns(y ~ x, data = df, prior_sigma = list(dist = "unknown", params = list())), "Supported prior distributions for sigma")
  expect_error(bnns(y ~ x, data = df, prior_sigma = list(dist = "half_normal")), "'prior_sigma' must contain 'dist' and 'params' elements.")
  
  # data validation
  expect_error(bnns(y ~ x), "Both 'formula' and 'data' must be provided.")
  expect_error(bnns(y ~ x, data = 1:10), "'data' must be a data.frame.")
  
  df_na <- data.frame(x = 1:10, y = c(NA, 2:10))
  expect_error(bnns(y ~ x, data = df_na), "'data' contains missing values.")
  
  df_inf <- data.frame(x = 1:10, y = c(Inf, 2:10))
  expect_error(bnns(y ~ x, data = df_inf), "'data' contains invalid values")
  
  expect_error(bnns_train(matrix(1:10), factor(rep("A", 10)), out_act_fn = 3), "train_y must have at least 3 levels")
  expect_error(bnns_train(matrix(1:10), 1:10, out_act_fn = 3), "train_y must be a factor")
  expect_error(bnns_train(matrix(1:10), 1:10, out_act_fn = 2), "train_y must have only 0/1 values")
})

test_that("bnns_parsnip_helpers coverage", {
  expect_equal(bnns:::translate_activation("sigmoid"), 2L)
  expect_equal(bnns:::translate_activation("relu"), 4L)
  expect_equal(bnns:::translate_activation("tanh"), 1L)
  expect_error(bnns:::translate_activation("unknown"), "Unknown activation")
  
  expect_equal(bnns:::detect_output_activation(1:10), 1L)
  expect_equal(bnns:::detect_output_activation(factor(c(0, 1, 0, 1))), 2L)
  expect_equal(bnns:::detect_output_activation(factor(c("A", "B", "C"))), 3L)
})

test_that("generate_stan_code coverage", {
  code <- bnns:::generate_stan_code(num_layers = 1, nodes = 2, out_act_fn = 1, prior_weights_dist = "horseshoe")
  expect_true(grepl("horseshoe", code) || grepl("lambda_w_out", code))
})