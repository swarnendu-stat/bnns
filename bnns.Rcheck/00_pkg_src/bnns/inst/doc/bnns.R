## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
# Install from CRAN (if available)
#  install.packages("bnns")

# Or install the development version from GitHub
# devtools::install_github("swarnendu-stat/bnns")

## -----------------------------------------------------------------------------
library(bnns)

## -----------------------------------------------------------------------------
# Generate training data
set.seed(123)
df <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))

## -----------------------------------------------------------------------------
# Binary classification response
df$y_bin <- sample(0:1, 10, replace = TRUE)

# Multiclass classification response
df$y_cat <- factor(sample(letters[1:3], 10, replace = TRUE)) # 3 classes

## ----message=FALSE, warning=FALSE, echo = TRUE, results = 'hide', eval = FALSE----
# model_reg <- bnns(
#   y ~ -1 + x1 + x2,
#   data = df,
#   L = 1, # Number of hidden layers
#   nodes = 2, # Nodes per layer
#   act_fn = 3, # Activation functions: 3 = ReLU
#   out_act_fn = 1, # Output activation function: 1 = Identity (for regression)
#   iter = 1e1,  # Very low number of iteration is shown, increase to at least 1e3 for meaningful inference
#   warmup = 5,  # Very low number of warmup is shown, increase to at least 2e2 for meaningful inference
#   chains = 1
# )

## ----message=FALSE, warning=FALSE, echo = TRUE, results = 'hide', eval = FALSE----
# model_bin <- bnns(
#   y_bin ~ -1 + x1 + x2,
#   data = df,
#   L = 1,
#   nodes = c(16),
#   act_fn = c(2),
#   out_act_fn = 2, # Output activation: 2 = Logistic sigmoid
#   iter = 2e2,
#   warmup = 1e2,
#   chains = 1
# )

## ----message=FALSE, warning=FALSE, echo = TRUE, results = 'hide', eval = FALSE----
# model_cat <- bnns(
#   y_cat ~ -1 + x1 + x2,
#   data = df,
#   L = 3,
#   nodes = c(32, 16, 8),
#   act_fn = c(3, 2, 2),
#   out_act_fn = 3, # Output activation: 3 = Softmax
#   iter = 2e2,
#   warmup = 1e2,
#   chains = 1
# )

## ----eval = FALSE-------------------------------------------------------------
# summary(model_reg)

## ----eval=FALSE---------------------------------------------------------------
# summary(model_bin)
# summary(model_cat)

## ----eval = FALSE-------------------------------------------------------------
# # New data
# test_x <- matrix(runif(10), nrow = 5, ncol = 2) |>
#   data.frame() |>
#   `colnames<-`(c("x1", "x2"))
# 
# # Regression predictions
# pred_reg <- predict(model_reg, test_x)

## ----eval = FALSE-------------------------------------------------------------
# # Binary classification predictions
# pred_bin <- predict(model_bin, test_x)
# 
# # Multiclass classification predictions
# pred_cat <- predict(model_cat, test_x)

## ----eval = FALSE-------------------------------------------------------------
# # True responses
# test_y <- rnorm(5)
# 
# # Evaluate predictions
# metrics_reg <- measure_cont(obs = test_y, pred = pred_reg)
# print(metrics_reg)

## ----eval = FALSE-------------------------------------------------------------
# # True responses
# test_y_bin <- sample(c(rep(0, 2), rep(1, 3)), 5)
# 
# # Evaluate predictions
# metrics_bin <- measure_bin(obs = test_y_bin, pred = pred_bin)

## ----eval = FALSE-------------------------------------------------------------
# # True responses
# test_y_cat <- factor(sample(letters[1:3], 5, replace = TRUE))
# 
# # Evaluate predictions
# metrics_cat <- measure_cat(obs = test_y_cat, pred = pred_cat)

## ----message=FALSE, warning=FALSE, echo = TRUE, results = 'hide', eval = FALSE----
# model_cat_cauchy <- bnns(
#   y_cat ~ -1 + x1 + x2,
#   data = df,
#   L = 3,
#   nodes = c(32, 16, 8),
#   act_fn = c(3, 2, 2),
#   out_act_fn = 3, # Output activation: 3 = Softmax
#   iter = 2e2,
#   warmup = 1e2,
#   chains = 1,
#   prior_weights = list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))
# )

## ----eval = FALSE-------------------------------------------------------------
# # Evaluate predictions
# metrics_cat_cauchy <- measure_cat(obs = test_y_cat, pred = predict(model_cat_cauchy, test_x))

