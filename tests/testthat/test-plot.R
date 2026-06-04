test_that("plot.bnns with type='pred_prob' works for classification", {
  skip_on_cran()
  
  # 1. Binary Classification Test
  df_bin <- data.frame(
    x1 = runif(20), 
    x2 = runif(20),
    y = factor(sample(c("Class_A", "Class_B"), 20, replace = TRUE))
  )
  
  fit_bin <- bnns(y ~ x1 + x2, data = df_bin, out_act_fn = 2, 
                  iter = 20, warmup = 10, chains = 1, refresh = 0)
  
  p_bin <- plot(fit_bin, type = "pred_prob")
  expect_s3_class(p_bin, "ggplot")
  
  # 2. Multiclass Classification Test
  df_multi <- data.frame(
    x1 = runif(30), 
    x2 = runif(30),
    y = factor(sample(c("Class_A", "Class_B", "Class_C"), 30, replace = TRUE))
  )
  
  fit_multi <- bnns(y ~ x1 + x2, data = df_multi, out_act_fn = 3, 
                    iter = 20, warmup = 10, chains = 1, refresh = 0)
  
  p_multi <- plot(fit_multi, type = "pred_prob")
  expect_s3_class(p_multi, "ggplot")
})

test_that("plot.bnns with type='pred_prob' fails for regression", {
  skip_on_cran()
  
  df_reg <- data.frame(x1 = runif(20), y = rnorm(20))
  fit_reg <- bnns(y ~ x1, data = df_reg, out_act_fn = 1, 
                  iter = 20, warmup = 10, chains = 1, refresh = 0)
  
  expect_error(plot(fit_reg, type = "pred_prob"), "only applicable for classification models")
})