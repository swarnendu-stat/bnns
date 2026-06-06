library(testthat)
library(bnns)

test_that("bnns prior validations work", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  
  # Invalid prior_weights structure
  expect_error(bnns(y ~ x1, data = df, prior_weights = list(dist = "normal")), "must be a list with elements 'dist' and 'params'")
  expect_error(bnns(y ~ x1, data = df, prior_weights = "normal"), "must be a list with elements 'dist' and 'params'")
  
  # Unsupported distribution
  expect_error(bnns(y ~ x1, data = df, prior_weights = list(dist = "unknown", params = list(mean = 0))), "Unsupported distribution for weights: unknown")
  
  # Horseshoe prior should not need params to be populated, but 'params' element must exist
  result <- suppressWarnings(bnns(y ~ x1, data = df, iter = 10, warmup = 5, chains = 1, prior_weights = list(dist = "horseshoe", params = list())))
  expect_s3_class(result, "bnns")
  
  # Prior bias unsupported dist
  expect_error(bnns(y ~ x1, data = df, prior_bias = list(dist = "unknown", params = list(mean = 0))), "Unsupported distribution for biases: unknown")
  expect_error(bnns(y ~ x1, data = df, prior_sigma = list(dist = "unknown", params = list(sd = 1))), "Supported prior distributions for sigma are 'half_normal' and 'inv_gamma'.")
})

test_that("bnns string activations are translated correctly", {
  df <- data.frame(x1 = runif(10), y = c(0, 1, sample(0:1, 8, replace=TRUE)))
  result <- suppressWarnings(bnns(y ~ x1, data = df, act_fn = "relu", out_act_fn = "sigmoid", iter = 10, warmup = 5, chains = 1))
  expect_equal(as.numeric(result$data$act_fn), 4)
  expect_equal(result$data$out_act_fn, 2)
})

test_that("bnns auto-corrects out_act_fn from 1 to 2 or 3 for factors", {
  df_bin <- data.frame(x1 = runif(10), y = factor(c("A", "B", sample(c("A","B"), 8, replace=TRUE))))
  result_bin <- suppressWarnings(bnns(y ~ x1, data = df_bin, out_act_fn = 1, iter = 10, warmup = 5, chains = 1))
  expect_equal(result_bin$data$out_act_fn, 2)
  
  df_multi <- data.frame(x1 = runif(10), y = factor(c("A", "B", "C", sample(c("A","B","C"), 7, replace=TRUE))))
  result_multi <- suppressWarnings(bnns(y ~ x1, data = df_multi, out_act_fn = 1, iter = 10, warmup = 5, chains = 1))
  expect_equal(result_multi$data$out_act_fn, 3)
})

test_that("bnns handles mismatched lengths of nodes or act_fn", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  expect_error(bnns(y ~ x1, data = df, L = 2, nodes = c(2, 3, 4)), "nodes must be of length L")
  expect_error(bnns(y ~ x1, data = df, L = 2, act_fn = c(1, 2, 3)), "act_fn must be of length L")
})

test_that("bnns.default works with invalid y type", {
  df <- data.frame(x1 = runif(10), y = letters[1:10]) # Character, not factor or numeric
  expect_error(bnns(y ~ x1, data = df), "response variable must be a numeric vector or a factor")
})

test_that("bnns parameter validation constraints", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  expect_error(bnns(y ~ x1, data = df, L = 0.5), "L must be a positive integer")
  expect_error(bnns(y ~ x1, data = df, nodes = -1), "nodes must be positive integer")
  expect_error(bnns(y ~ x1, data = df, act_fn = 10), "act_fn must be a sequence of 1/2/3/4/5")
  expect_error(bnns(y ~ x1, data = df, out_act_fn = 4), "out_act_fn must be 1, 2, or 3")
  
  expect_error(bnns:::bnns_train(), "Argument train_x is missing")
  expect_error(bnns:::bnns_train(train_x = matrix(runif(10))), "Argument train_y is missing")
})

test_that("bnns handles missing/invalid values", {
  df <- data.frame(x1 = c(1, NA, 3), y = rnorm(3))
  expect_error(bnns(y ~ x1, data = df), "contains missing values")

  df2 <- data.frame(x1 = c(1, Inf, 3), y = rnorm(3))
  expect_error(bnns(y ~ x1, data = df2), "contains invalid values")
})

test_that("bnns empty args", {
  expect_error(bnns(data = data.frame(y=1)), "Both 'formula' and 'data' must be provided.")
  expect_error(bnns(y ~ 1), "Both 'formula' and 'data' must be provided.")
  expect_invisible(bnns())
  expect_error(bnns(y ~ x1, data = list(x1=1, y=2)), "'data' must be a data.frame.")
})

test_that("bnns_params functions work", {
  expect_s3_class(L(), "quant_param")
  expect_s3_class(warmup(), "quant_param")
  expect_s3_class(chains(), "quant_param")
  expect_s3_class(iter(), "quant_param")
  expect_s3_class(nodes(), "quant_param")
  expect_s3_class(act_fn(), "qual_param")
})

test_that("predict functions for parsnip work", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  mod_reg <- suppressWarnings(bnns(y ~ x1, data = df, iter = 10, warmup = 5, chains = 1))
  
  p_obj <- list(fit = mod_reg)
  pred_num <- predict_numeric_bnns(p_obj, new_data = df)
  expect_s3_class(pred_num, "tbl_df")
  expect_named(pred_num, ".pred")
  
  df_bin <- data.frame(x1 = runif(10), y = c(0, 1, sample(0:1, 8, replace=TRUE)))
  mod_bin <- suppressWarnings(bnns(y ~ x1, data = df_bin, out_act_fn = 2, iter = 10, warmup = 5, chains = 1))
  p_obj_bin <- list(fit = mod_bin, lvl = c("0", "1"))
  pred_bin_prob <- predict_prob_bnns(p_obj_bin, new_data = df_bin)
  expect_s3_class(pred_bin_prob, "tbl_df")
  expect_named(pred_bin_prob, c("0", "1"))
  
  pred_bin_class <- predict_class_bnns(p_obj_bin, new_data = df_bin)
  expect_s3_class(pred_bin_class, "tbl_df")
  expect_named(pred_bin_class, ".pred_class")

  df_cat <- data.frame(x1 = runif(10), y = factor(sample(c("A","B","C"), 10, replace=TRUE)))
  mod_cat <- suppressWarnings(bnns(y ~ x1, data = df_cat, out_act_fn = 3, iter = 10, warmup = 5, chains = 1))
  p_obj_cat <- list(fit = mod_cat, lvl = c("A", "B", "C"))
  pred_cat_prob <- predict_prob_bnns(p_obj_cat, new_data = df_cat)
  expect_s3_class(pred_cat_prob, "tbl_df")
  expect_named(pred_cat_prob, c("A", "B", "C"))
})

test_that("plot functions run", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  mod <- suppressWarnings(bnns(y ~ x1, data = df, iter = 10, warmup = 5, chains = 1))
  
  p_trace <- plot(mod, type = "trace")
  expect_s3_class(p_trace, "ggplot")
  
  p_dens <- plot(mod, type = "density")
  expect_s3_class(p_dens, "ggplot")
  
  p_ppc <- plot(mod, type = "posterior_predictive")
  expect_s3_class(p_ppc, "ggplot")
  
  expect_error(plot(mod, type = "pred_prob"), "only applicable for classification models")
})

test_that("pred_prob plot runs for classification", {
  df_bin <- data.frame(x1 = runif(10), y = c(0, 1, sample(0:1, 8, replace=TRUE)))
  mod_bin <- suppressWarnings(bnns(y ~ x1, data = df_bin, out_act_fn = 2, iter = 10, warmup = 5, chains = 1))
  
  p_bin <- plot(mod_bin, type = "pred_prob")
  expect_s3_class(p_bin, "ggplot")
  
  df_cat <- data.frame(x1 = runif(10), y = factor(sample(c("A","B","C"), 10, replace=TRUE)))
  mod_cat <- suppressWarnings(bnns(y ~ x1, data = df_cat, out_act_fn = 3, iter = 10, warmup = 5, chains = 1))
  
  p_cat <- plot(mod_cat, type = "pred_prob")
  expect_s3_class(p_cat, "ggplot")
})

test_that("opencl_diagnostics runs", {
  out <- capture.output(suppressMessages(opencl_diagnostics()))
  expect_type(out, "character")
})

test_that("loo and waic run", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  mod <- suppressWarnings(bnns(y ~ x1, data = df, iter = 20, warmup = 10, chains = 1))
  
  l <- suppressWarnings(loo.bnns(mod))
  expect_s3_class(l, "psis_loo")
  
  w <- suppressWarnings(waic.bnns(mod))
  expect_s3_class(w, "waic")
})

test_that("Validate prior parameters logic", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  
  expect_error(bnns(y ~ x1, data = df, prior_weights = list(dist = "normal", params = list(mean = 0))), "For 'normal' distribution, 'params' must contain 'mean' and 'sd'")
  expect_error(bnns(y ~ x1, data = df, prior_weights = list(dist = "uniform", params = list(alpha = 0))), "For 'uniform' distribution, 'params' must contain 'alpha' and 'beta'")
  expect_error(bnns(y ~ x1, data = df, prior_weights = list(dist = "cauchy", params = list(mu = 0))), "For 'cauchy' distribution, 'params' must contain 'mu' and 'sigma'")
  
  expect_error(bnns(y ~ x1, data = df, prior_sigma = list(dist = "half_normal", params = list(mean = 0))), "For 'half_normal' distribution, 'params' must contain 'mean' and 'sd'")
  expect_error(bnns(y ~ x1, data = df, prior_sigma = list(dist = "inv_gamma", params = list(alpha = 0))), "For 'inv_gamma' distribution, 'params' must contain 'alpha' and 'beta'")
})

test_that("backend warnings and non-normalized checks", {
  df <- data.frame(x1 = runif(10), y = rnorm(10))
  
  expect_warning(
    expect_error(bnns(y ~ x1, data = df, backend = "rstan", use_gpu = TRUE, iter = 10, warmup = 5, chains = 1, L = 0.5),
                 "L must be a positive integer"),
    "GPU acceleration is only supported with the 'cmdstanr' backend"
  )
  expect_warning(
    expect_error(bnns(y ~ x1, data = df, backend = "cmdstanr", algorithm = "HMC", iter = 10, warmup = 5, chains = 1, L = 0.5),
                 "L must be a positive integer"),
    "The 'cmdstanr' backend does not natively expose static HMC"
  )
  
  res <- suppressWarnings(bnns(y ~ x1, data = df, normalize = FALSE, iter = 10, warmup = 5, chains = 1))
  expect_s3_class(res, "bnns")
})
