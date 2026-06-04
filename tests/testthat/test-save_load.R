test_that("save_bnns and load_bnns work correctly", {
  skip_on_cran()
  
  df <- data.frame(x = runif(20), y = rnorm(20))
  fit <- bnns(y ~ x, data = df, L = 1, nodes = 2, out_act_fn = 1, iter = 20, warmup = 10, chains = 1, refresh = 0)
  
  tmp <- tempfile(fileext = ".rds")
  
  # Test saving
  save_bnns(fit, tmp)
  expect_true(file.exists(tmp))
  
  # Test loading
  fit_loaded <- load_bnns(tmp)
  expect_s3_class(fit_loaded, "bnns")
  
  # Make sure predictions are perfectly identical
  expect_equal(predict(fit, type = "mean"), predict(fit_loaded, type = "mean"))
  
  # Cleanup
  unlink(tmp)
})