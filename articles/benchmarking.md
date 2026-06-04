# Benchmarking bnns

``` r

# Ensuring required packages are installed before proceeding
stopifnot("mlbench not installed" = requireNamespace("mlbench", quietly = TRUE))
stopifnot("rsample not installed" = requireNamespace("rsample", quietly = TRUE))
stopifnot("ranger not installed" = requireNamespace("ranger", quietly = TRUE))

# Loading the necessary libraries
library(bnns)    # For Bayesian Neural Networks
library(mlbench) # For benchmark datasets
library(rsample) # For data splitting
library(ranger)  # For random forest model comparison
set.seed(123)    # Setting seed for reproducibility
```

## Introduction

This article demonstrates the performance of the `bnns` package on three
datasets from the `mlbench` package:

- **Regression**: `mlbench.friedman1` dataset  
- **Binary Classification**: `mlbench.spirals` dataset  
- **Multi-class Classification**: `mlbench.waveform` dataset

For each dataset, we:

1.  Prepare the data for training and testing.
2.  Build a Bayesian Neural Network using the `bnns` package.
3.  Evaluate the model’s predictive performance.
4.  To compare, show the performance of the randomforest algorithm from
    `ranger` package with default settings.

------------------------------------------------------------------------

## Regression: Friedman1 Dataset

### Dataset Description

The dataset generated with `mlbench.friedman1` is the regression problem
Friedman 1 as described in Friedman (1991) and Breiman (1996). Inputs
are 10 independent variables uniformly distributed on the interval
\[0,1\], only 5 out of these 10 are actually used. Outputs are created
according to the formula

``` math
y = 10 \sin(\pi x_1 x_2) + 20 (x_3 - 0.5)^2 + 10 x_4 + 5 x_5 + \epsilon
```

$`\epsilon \sim N(0, 1).`$

``` r

# Generating the Friedman1 dataset
friedman1_data <- mlbench.friedman1(n = 100, sd = 1)

# Splitting the data into training (80%) and testing (20%) sets
friedman1_split <- initial_split(
  cbind.data.frame(y = friedman1_data$y, friedman1_data$x),
  prop = 0.8
)
friedman1_train <- training(friedman1_split)  # Training data
friedman1_test <- testing(friedman1_split)   # Testing data
```

### Model Training

``` r

# Training a Bayesian Neural Network with a single hidden layer and 4 nodes
friedman1_bnn <- bnns(y ~ -1 + .,
  data = friedman1_train, L = 1, nodes = 4, act_fn = 3,
  out_act_fn = 1, iter = 1e3, warmup = 2e2
)
#> Trying to compile a simple C file
#> Running /opt/R/4.6.0/lib/R/bin/R CMD SHLIB foo.c
#> using C compiler: ‘gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0’
#> gcc -std=gnu2x -I"/opt/R/4.6.0/lib/R/include" -DNDEBUG   -I"/home/runner/work/_temp/Library/Rcpp/include/"  -I"/home/runner/work/_temp/Library/RcppEigen/include/"  -I"/home/runner/work/_temp/Library/RcppEigen/include/unsupported"  -I"/home/runner/work/_temp/Library/BH/include" -I"/home/runner/work/_temp/Library/StanHeaders/include/src/"  -I"/home/runner/work/_temp/Library/StanHeaders/include/"  -I"/home/runner/work/_temp/Library/RcppParallel/include/"  -I"/home/runner/work/_temp/Library/rstan/include" -DEIGEN_NO_DEBUG  -DBOOST_DISABLE_ASSERTS  -DBOOST_PENDING_INTEGER_LOG2_HPP  -DSTAN_THREADS  -DUSE_STANC3 -DSTRICT_R_HEADERS  -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION  -D_HAS_AUTO_PTR_ETC=0  -include '/home/runner/work/_temp/Library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp'  -D_REENTRANT -DRCPP_PARALLEL_USE_TBB=1   -I/usr/local/include    -fpic  -g -O2  -c foo.c -o foo.o
#> In file included from /home/runner/work/_temp/Library/RcppEigen/include/Eigen/Core:19,
#>                  from /home/runner/work/_temp/Library/RcppEigen/include/Eigen/Dense:1,
#>                  from /home/runner/work/_temp/Library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22,
#>                  from <command-line>:
#> /home/runner/work/_temp/Library/RcppEigen/include/Eigen/src/Core/util/Macros.h:679:10: fatal error: cmath: No such file or directory
#>   679 | #include <cmath>
#>       |          ^~~~~~~
#> compilation terminated.
#> make: *** [/opt/R/4.6.0/lib/R/etc/Makeconf:190: foo.o] Error 1
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
#> Chain 1: 
#> Chain 1: Gradient evaluation took 0.000115 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 1.15 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
#> Chain 2: 
#> Chain 2: Gradient evaluation took 0.000113 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 1.13 seconds.
#> Chain 2: Adjust your expectations accordingly!
#> Chain 2: 
#> Chain 2: 
#> Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
#> Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 2: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 1: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 2: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 1: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 2: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 2: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 0.912 seconds (Warm-up)
#> Chain 2:                4.736 seconds (Sampling)
#> Chain 2:                5.648 seconds (Total)
#> Chain 2: 
#> Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 1.092 seconds (Warm-up)
#> Chain 1:                5.133 seconds (Sampling)
#> Chain 1:                6.225 seconds (Total)
#> Chain 1:
```

### Model Evaluation

``` r

# Making predictions on the test set and evaluating model performance
friedman1_bnn_pred <- predict(friedman1_bnn, friedman1_test, type = "mean")
bnns::measure_cont(friedman1_test$y, friedman1_bnn_pred)  # Measures like RMSE, MAE
#> $rmse
#> [1] 2.331574
#> 
#> $mae
#> [1] 1.913072
```

### Model Comparison

``` r

# Training a random forest model for comparison
friedman1_rf <- ranger(
  y ~ -1 + .,
  data = friedman1_train |> `colnames<-`(c("y", paste0("x", 1:10)))
)

# Making predictions with random forest and evaluating performance
friedman1_rf_pred <- predict(
  friedman1_rf,
  friedman1_test |> `colnames<-`(c("y", paste0("x", 1:10)))
)
measure_cont(friedman1_test$y, friedman1_rf_pred$predictions)
#> $rmse
#> [1] 3.099387
#> 
#> $mae
#> [1] 2.517299
```

------------------------------------------------------------------------

## Binary Classification: Spirals Dataset

### Dataset Description

The dataset generated with the `mlbench.spirals` consists of points on
two entangled spirals. If `sd>0`, then Gaussian noise is added to each
data point.

``` r

# Generating the Spirals dataset with Gaussian noise
spirals_data <- mlbench.spirals(100, 1.5, 0.05)
spirals_data <- cbind.data.frame(y = spirals_data$classes, spirals_data$x) |>
  transform(y = as.numeric(y) - 1)  # Converting to binary 0/1

# Splitting the data into training and testing sets (stratified by class)
spirals_split <- initial_split(spirals_data, prop = 0.8, strata = "y")
spirals_train <- training(spirals_split)  # Training data
spirals_test <- testing(spirals_split)   # Testing data
```

### Model Training

``` r

# Training a Bayesian Neural Network with three hidden layers
spirals_bnn <- bnns(y ~ -1 + .,
  data = spirals_train, L = 3,
  nodes = c(64, 64, 16), act_fn = c(1, 4, 4),
  out_act_fn = 2, iter = 1e3, warmup = 2e2
)
#> Trying to compile a simple C file
#> Running /opt/R/4.6.0/lib/R/bin/R CMD SHLIB foo.c
#> using C compiler: ‘gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0’
#> gcc -std=gnu2x -I"/opt/R/4.6.0/lib/R/include" -DNDEBUG   -I"/home/runner/work/_temp/Library/Rcpp/include/"  -I"/home/runner/work/_temp/Library/RcppEigen/include/"  -I"/home/runner/work/_temp/Library/RcppEigen/include/unsupported"  -I"/home/runner/work/_temp/Library/BH/include" -I"/home/runner/work/_temp/Library/StanHeaders/include/src/"  -I"/home/runner/work/_temp/Library/StanHeaders/include/"  -I"/home/runner/work/_temp/Library/RcppParallel/include/"  -I"/home/runner/work/_temp/Library/rstan/include" -DEIGEN_NO_DEBUG  -DBOOST_DISABLE_ASSERTS  -DBOOST_PENDING_INTEGER_LOG2_HPP  -DSTAN_THREADS  -DUSE_STANC3 -DSTRICT_R_HEADERS  -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION  -D_HAS_AUTO_PTR_ETC=0  -include '/home/runner/work/_temp/Library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp'  -D_REENTRANT -DRCPP_PARALLEL_USE_TBB=1   -I/usr/local/include    -fpic  -g -O2  -c foo.c -o foo.o
#> In file included from /home/runner/work/_temp/Library/RcppEigen/include/Eigen/Core:19,
#>                  from /home/runner/work/_temp/Library/RcppEigen/include/Eigen/Dense:1,
#>                  from /home/runner/work/_temp/Library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22,
#>                  from <command-line>:
#> /home/runner/work/_temp/Library/RcppEigen/include/Eigen/src/Core/util/Macros.h:679:10: fatal error: cmath: No such file or directory
#>   679 | #include <cmath>
#>       |          ^~~~~~~
#> compilation terminated.
#> make: *** [/opt/R/4.6.0/lib/R/etc/Makeconf:190: foo.o] Error 1
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
#> Chain 1: 
#> Chain 1: Gradient evaluation took 0.001948 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 19.48 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 2: 
#> Chain 2: Gradient evaluation took 0.001972 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 19.72 seconds.
#> Chain 2: Adjust your expectations accordingly!
#> Chain 2: 
#> Chain 2: 
#> Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
#> Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
#> Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 2: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 1: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 1: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 2: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 1: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 2: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 156.121 seconds (Warm-up)
#> Chain 1:                662.712 seconds (Sampling)
#> Chain 1:                818.833 seconds (Total)
#> Chain 1: 
#> Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 155.924 seconds (Warm-up)
#> Chain 2:                664.048 seconds (Sampling)
#> Chain 2:                819.972 seconds (Total)
#> Chain 2:
```

### Model Evaluation

``` r

# Making predictions and calculating binary classification metrics (e.g., AUC)
spirals_bnn_pred <- predict(spirals_bnn, spirals_test, type = "mean")
measure_bin(spirals_test$y, spirals_bnn_pred)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> $conf_mat
#>    pred_label
#> obs  0  1
#>   0 10  0
#>   1  1  9
#> 
#> $accuracy
#> [1] 0.95
#> 
#> $ROC
#> 
#> Call:
#> roc.default(response = obs, predictor = pred)
#> 
#> Data: pred in 10 controls (obs 0) < 10 cases (obs 1).
#> Area under the curve: 0.98
#> 
#> $AUC
#> [1] 0.98
```

### Model Comparison

``` r

# Training a random forest model for comparison
spirals_rf <- ranger(
  y ~ -1 + .,
  data = spirals_train |> `colnames<-`(c("y", paste0("x", 1:2)))
)

# Evaluating the random forest model
spirals_rf_pred <- predict(
  spirals_rf,
  spirals_test |> `colnames<-`(c("y", paste0("x", 1:2)))
)
measure_bin(spirals_test$y, spirals_rf_pred$predictions)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> $conf_mat
#>    pred_label
#> obs 0 1
#>   0 5 5
#>   1 4 6
#> 
#> $accuracy
#> [1] 0.55
#> 
#> $ROC
#> 
#> Call:
#> roc.default(response = obs, predictor = pred)
#> 
#> Data: pred in 10 controls (obs 0) < 10 cases (obs 1).
#> Area under the curve: 0.64
#> 
#> $AUC
#> [1] 0.64
```

------------------------------------------------------------------------

## Multi-class Classification: Waveform Dataset

### Dataset Description

The dataset generated with `mlbench.waveform` consists of 21 attributes
with continuous values and a variable showing the 3 classes (33% for
each of 3 classes). Each class is generated from a combination of 2 of 3
“base” waves.

``` r

# Generating the Waveform dataset
waveform_data <- mlbench.waveform(100)
waveform_data <- cbind.data.frame(y = waveform_data$classes, waveform_data$x) |>
  transform(y = as.factor(y))  # Converting the target to a factor

# Splitting the data into training and testing sets (stratified by class)
waveform_split <- initial_split(waveform_data, prop = 0.8, strata = "y")
waveform_train <- training(waveform_split)  # Training data
waveform_test <- testing(waveform_split)   # Testing data
```

### Model Training

``` r

# Training a Bayesian Neural Network with two hidden layers
waveform_bnn <- bnns(y ~ -1 + .,
  data = waveform_train, L = 2, nodes = c(2, 2),
  act_fn = 2:3, out_act_fn = 3, iter = 1e3, warmup = 2e2
)
#> Trying to compile a simple C file
#> Running /opt/R/4.6.0/lib/R/bin/R CMD SHLIB foo.c
#> using C compiler: ‘gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0’
#> gcc -std=gnu2x -I"/opt/R/4.6.0/lib/R/include" -DNDEBUG   -I"/home/runner/work/_temp/Library/Rcpp/include/"  -I"/home/runner/work/_temp/Library/RcppEigen/include/"  -I"/home/runner/work/_temp/Library/RcppEigen/include/unsupported"  -I"/home/runner/work/_temp/Library/BH/include" -I"/home/runner/work/_temp/Library/StanHeaders/include/src/"  -I"/home/runner/work/_temp/Library/StanHeaders/include/"  -I"/home/runner/work/_temp/Library/RcppParallel/include/"  -I"/home/runner/work/_temp/Library/rstan/include" -DEIGEN_NO_DEBUG  -DBOOST_DISABLE_ASSERTS  -DBOOST_PENDING_INTEGER_LOG2_HPP  -DSTAN_THREADS  -DUSE_STANC3 -DSTRICT_R_HEADERS  -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION  -D_HAS_AUTO_PTR_ETC=0  -include '/home/runner/work/_temp/Library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp'  -D_REENTRANT -DRCPP_PARALLEL_USE_TBB=1   -I/usr/local/include    -fpic  -g -O2  -c foo.c -o foo.o
#> In file included from /home/runner/work/_temp/Library/RcppEigen/include/Eigen/Core:19,
#>                  from /home/runner/work/_temp/Library/RcppEigen/include/Eigen/Dense:1,
#>                  from /home/runner/work/_temp/Library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22,
#>                  from <command-line>:
#> /home/runner/work/_temp/Library/RcppEigen/include/Eigen/src/Core/util/Macros.h:679:10: fatal error: cmath: No such file or directory
#>   679 | #include <cmath>
#>       |          ^~~~~~~
#> compilation terminated.
#> make: *** [/opt/R/4.6.0/lib/R/etc/Makeconf:190: foo.o] Error 1
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
#> Chain 1: 
#> Chain 1: Gradient evaluation took 0.000177 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 1.77 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
#> Chain 2: 
#> Chain 2: Gradient evaluation took 0.000166 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 1.66 seconds.
#> Chain 2: Adjust your expectations accordingly!
#> Chain 2: 
#> Chain 2: 
#> Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
#> Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 1: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 2: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 1: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 2: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 1: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 2: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 0.768 seconds (Warm-up)
#> Chain 1:                2.4 seconds (Sampling)
#> Chain 1:                3.168 seconds (Total)
#> Chain 1: 
#> Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 0.799 seconds (Warm-up)
#> Chain 2:                2.374 seconds (Sampling)
#> Chain 2:                3.173 seconds (Total)
#> Chain 2:
```

### Model Evaluation

``` r

# Making predictions and evaluating multi-class classification metrics
waveform_bnn_pred <- predict(waveform_bnn, waveform_test)
measure_cat(waveform_test$y, waveform_bnn_pred)
#> $log_loss
#> [1] 0.5167825
#> 
#> $ROC
#> 
#> Call:
#> multiclass.roc.default(response = obs, predictor = `colnames<-`(data.frame(pred),     levels(obs)))
#> 
#> Data: multivariate predictor `colnames<-`(data.frame(pred), levels(obs)) with 3 levels of obs: 1, 2, 3.
#> Multi-class area under the curve: 0.9777
#> 
#> $AUC
#> [1] 0.9776786
```

### Model Comparison

``` r

# Training a random forest model with probability outputs for comparison
waveform_rf <- ranger(
  y ~ -1 + .,
  data = waveform_train |> `colnames<-`(c("y", paste0("x", 1:21))),
  probability = TRUE
)

# Evaluating the random forest model
waveform_rf_pred <- predict(
  waveform_rf,
  waveform_test |> `colnames<-`(c("y", paste0("x", 1:21)))
)
measure_cat(waveform_test$y, waveform_rf_pred$predictions)
#> $log_loss
#>         1 
#> 0.5073519 
#> 
#> $ROC
#> 
#> Call:
#> multiclass.roc.default(response = obs, predictor = `colnames<-`(data.frame(pred),     levels(obs)))
#> 
#> Data: multivariate predictor `colnames<-`(data.frame(pred), levels(obs)) with 3 levels of obs: 1, 2, 3.
#> Multi-class area under the curve: 0.9742
#> 
#> $AUC
#> [1] 0.9742063
```

------------------------------------------------------------------------

## Summary

The `bnns` package showcases strong predictive performance across
regression, binary classification, and multi-class classification tasks.
In addition to accurate predictions, it provides posterior
distributions, enabling:

1.  Uncertainty Quantification: Offers insights into the confidence of
    predictions, crucial for high-stakes applications like clinical
    trials and finance.
2.  Probabilistic Decision-Making: Facilitates decisions under
    uncertainty by integrating Bayesian principles.
3.  Model Comparisons: Demonstrates comparable performance to the ranger
    package, with the added advantage of interpretability through
    Bayesian inference.

Overall, `bnns` is a powerful tool for tasks requiring both predictive
accuracy and interpretability, making it suitable for various domains.
