## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = requireNamespace("tidymodels", quietly = TRUE)
)

## ----setup, message=FALSE, warning=FALSE--------------------------------------
library(tidymodels)
library(bnns)

## ----reg-spec-----------------------------------------------------------------
bnn_reg_spec <- mlp(
  mode = "regression",
  hidden_units = 5,
  epochs = 500,
  activation = "relu"
) %>% 
  set_engine(
    engine = "bnns", 
    chains = 2, 
    warmup = 250, 
    refresh = 0,
    seed = 123
  )

bnn_reg_spec

## ----reg-fit, eval=FALSE------------------------------------------------------
# bnn_reg_wf <- workflow() %>%
#   add_model(bnn_reg_spec) %>%
#   add_formula(mpg ~ hp + wt + cyl + disp)
# 
# # Fit the model
# bnn_reg_fit <- fit(bnn_reg_wf, data = mtcars)
# 
# bnn_reg_fit

## ----reg-pred, eval=FALSE-----------------------------------------------------
# predictions <- predict(bnn_reg_fit, new_data = mtcars)
# head(predictions)

## ----class-spec---------------------------------------------------------------
bnn_class_spec <- mlp(
  mode = "classification",
  hidden_units = 4,
  epochs = 500,
  activation = "tanh"
) %>% 
  set_engine(
    engine = "bnns", 
    chains = 1, 
    warmup = 200, 
    refresh = 0,
    seed = 456
  )

## ----class-fit, eval=FALSE----------------------------------------------------
# iris_rec <- recipe(Species ~ ., data = iris) %>%
#   step_normalize(all_numeric_predictors())
# 
# bnn_class_wf <- workflow() %>%
#   add_model(bnn_class_spec) %>%
#   add_recipe(iris_rec)
# 
# bnn_class_fit <- fit(bnn_class_wf, data = iris)

## ----class-pred, eval=FALSE---------------------------------------------------
# # 1. Predict hard classes (returns a .pred_class factor column)
# class_preds <- predict(bnn_class_fit, new_data = iris, type = "class")
# head(class_preds)
# 
# # 2. Predict class probabilities (returns .pred_{Level} columns)
# prob_preds <- predict(bnn_class_fit, new_data = iris, type = "prob")
# head(prob_preds)

## ----eval-metrics, eval=FALSE-------------------------------------------------
# eval_data <- bind_cols(iris, class_preds, prob_preds)
# 
# accuracy(eval_data, truth = Species, estimate = .pred_class)
# roc_auc(eval_data, truth = Species, .pred_setosa:.pred_virginica)

