test_that("parsnip engine registration is successful", {
  skip_if_not_installed("parsnip")
  
  engs <- parsnip::show_engines("mlp")
  expect_true("bnns" %in% engs$engine)
})