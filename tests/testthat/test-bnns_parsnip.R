test_that("parsnip engine registration works", {
  skip_if_not_installed("parsnip")
  
  # Check that "bnns" engine is registered for mlp in regression
  engines <- parsnip::show_engines("mlp")
  expect_true("bnns" %in% engines$engine)
  
  # Build a specification to test args mapping
  mlp_spec <- parsnip::mlp(
    mode = "regression",
    engine = "bnns",
    hidden_units = 10,
    epochs = 50,
    activation = "relu"
  )
  
  expect_s3_class(mlp_spec, "model_spec")
  expect_equal(mlp_spec$engine, "bnns")
})

test_that("registration can be forced for a new model to ensure coverage", {
  skip_if_not_installed("parsnip")
  
  model_name <- basename(tempfile("bnns_mlp_test_"))
  parsnip::set_new_model(model_name)
  parsnip::set_model_mode(model = model_name, mode = "regression")
  parsnip::set_model_mode(model = model_name, mode = "classification")
  expect_silent(bnns:::register_bnns_parsnip(force = TRUE, model = model_name))
  
  engines <- parsnip::show_engines(model_name)
  expect_true("bnns" %in% engines$engine)
})

test_that("parsnip helpers work correctly", {
  expect_equal(bnns:::translate_activation("tanh"), 1L)
  expect_equal(bnns:::translate_activation("sigmoid"), 2L)
  expect_equal(bnns:::translate_activation("softplus"), 3L)
  expect_equal(bnns:::translate_activation("relu"), 4L)
  expect_equal(bnns:::translate_activation("linear"), 5L)
  expect_error(bnns:::translate_activation("unknown_act"))
  expect_equal(bnns:::translate_activation(2), 2L)
  
  expect_equal(bnns:::detect_output_activation(rnorm(10)), 1L)
  expect_equal(bnns:::detect_output_activation(factor(c("A", "B"))), 2L)
  expect_equal(bnns:::detect_output_activation(factor(c("A", "B", "C"))), 3L)
})

test_that("parsnip post prediction helpers work correctly", {
  # Test numeric
  results_num <- matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)
  res_num <- bnns:::post_pred_numeric(results_num, NULL)
  expect_s3_class(res_num, "tbl_df")
  expect_equal(names(res_num), ".pred")
  expect_equal(res_num$.pred, c(1.5, 3.5))
  
  # Test binary class
  results_bin <- matrix(c(0.2, 0.4, 0.8, 0.9), nrow = 2, byrow = TRUE)
  mock_obj_bin <- list(fit = list(levels = c("No", "Yes")))
  res_class <- bnns:::post_pred_class(results_bin, mock_obj_bin)
  expect_s3_class(res_class, "tbl_df")
  expect_equal(names(res_class), ".pred_class")
  expect_equal(as.character(res_class$.pred_class), c("No", "Yes"))
  
  # Test binary prob
  res_prob <- bnns:::post_pred_prob(results_bin, mock_obj_bin)
  expect_s3_class(res_prob, "tbl_df")
  expect_equal(names(res_prob), c(".pred_No", ".pred_Yes"))
  expect_equal(res_prob$.pred_Yes, c(0.3, 0.85))
  expect_equal(res_prob$.pred_No, c(0.7, 0.15))
  
  # Test multiclass prob
  results_multi <- array(
    c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1),
    dim = c(2, 2, 3) # N=2, samples=2, classes=3
  )
  mock_obj_multi <- list(fit = list(levels = c("A", "B", "C")))
  res_multi_prob <- bnns:::post_pred_prob(results_multi, mock_obj_multi)
  expect_s3_class(res_multi_prob, "tbl_df")
  expect_equal(names(res_multi_prob), c(".pred_A", ".pred_B", ".pred_C"))
  expect_equal(res_multi_prob$.pred_A, c(0.2, 0.3))
  expect_equal(res_multi_prob$.pred_B, c(0.55, 0.5))
  expect_equal(res_multi_prob$.pred_C, c(0.15, 0.1))
})

test_that("parsnip models can be fitted and predicted", {
  skip_on_cran()
  skip_if_not_installed("parsnip")
  
  # 1. Regression
  df_reg <- data.frame(x1 = rnorm(5), x2 = rnorm(5), y = rnorm(5))
  
  spec_reg <- parsnip::set_engine(
    parsnip::mlp(
      mode = "regression",
      engine = "bnns",
      hidden_units = 2,
      epochs = 10
    ), 
    "bnns", chains = 1, warmup = 5, refresh = 0, backend = "rstan"
  )
  
  # Warning might appear due to low iterations, which is fine
  fit_reg <- suppressWarnings(parsnip::fit(spec_reg, y ~ x1 + x2, data = df_reg))
  
  pred_num <- predict(fit_reg, new_data = df_reg)
  expect_s3_class(pred_num, "tbl_df")
  expect_equal(names(pred_num), ".pred")
  
  pred_raw <- predict(fit_reg, new_data = df_reg, type = "raw")
  expect_true(is.matrix(pred_raw) || is.array(pred_raw))
  
  # 2. Classification (Binary)
  df_bin <- data.frame(x1 = rnorm(5), x2 = rnorm(5), y = factor(c("A", "B", sample(c("A", "B"), 3, replace = TRUE))))
  
  spec_class <- parsnip::set_engine(
    parsnip::mlp(
      mode = "classification",
      engine = "bnns",
      hidden_units = 2,
      epochs = 10
    ), 
    "bnns", chains = 1, warmup = 5, refresh = 0, backend = "rstan"
  )
  
  fit_bin <- suppressWarnings(parsnip::fit(spec_class, y ~ x1 + x2, data = df_bin))
  
  pred_class_bin <- predict(fit_bin, new_data = df_bin, type = "class")
  expect_s3_class(pred_class_bin, "tbl_df")
  expect_equal(names(pred_class_bin), ".pred_class")
  
  pred_prob_bin <- predict(fit_bin, new_data = df_bin, type = "prob")
  expect_s3_class(pred_prob_bin, "tbl_df")
  expect_equal(names(pred_prob_bin), c(".pred_A", ".pred_B"))
  
  # 3. Classification (Multiclass)
  df_multi <- data.frame(x1 = rnorm(10), x2 = rnorm(10), y = factor(c("A", "B", "C", sample(c("A", "B", "C"), 7, replace = TRUE))))
  
  fit_multi <- suppressWarnings(parsnip::fit(spec_class, y ~ x1 + x2, data = df_multi))
  
  pred_class_multi <- predict(fit_multi, new_data = df_multi, type = "class")
  expect_s3_class(pred_class_multi, "tbl_df")
  expect_equal(names(pred_class_multi), ".pred_class")
  
  pred_prob_multi <- predict(fit_multi, new_data = df_multi, type = "prob")
  expect_s3_class(pred_prob_multi, "tbl_df")
  expect_equal(names(pred_prob_multi), c(".pred_A", ".pred_B", ".pred_C"))
})

test_that(".onLoad executes properly", {
  expect_silent(bnns:::.onLoad(libname = "bnns", pkgname = "bnns"))
})