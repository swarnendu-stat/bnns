# Common Application Areas of bnns

``` r

stopifnot("mlbench not installed" = requireNamespace("mlbench", quietly = TRUE))
stopifnot("rsample not installed" = requireNamespace("rsample", quietly = TRUE))
library(bnns)
library(mlbench)
library(rsample)
set.seed(123)
```

## Introduction

This article demonstrates the use of the `bnns` package on three
datasets from the `mlbench` package:

- **Regression**: `BostonHousing` dataset  
- **Binary Classification**: `PimaIndiansDiabetes` dataset  
- **Multi-class Classification**: `Glass` dataset

For each dataset, we: 1. Prepare the data for training and testing. 2.
Build a Bayesian Neural Network using the `bnns` package. 3. Evaluate
the model’s predictive performance.

------------------------------------------------------------------------

## Regression: BostonHousing Dataset

### Dataset Description

The `BostonHousing` dataset contains information on housing prices in
Boston, with features like crime rate, average number of rooms, and
more.

``` r

data(BostonHousing)
BH_data <- BostonHousing
# Splitting data into training and testing sets
BH_split <- initial_split(BH_data, prop = 0.8)
BH_train <- training(BH_split)
BH_test <- testing(BH_split)
```

### Model Training

``` r

model_reg <- bnns(
  medv ~ -1 + .,
  data = BH_train, L = 2, out_act_fn = 1,
  iter = 1e3, warmup = 2e2, chains = 2, cores = 2
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
#> Chain 2: 
#> Chain 2: Gradient evaluation took 0.000267 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 2.67 seconds.
#> Chain 2: Adjust your expectations accordingly!
#> Chain 2: 
#> Chain 2: 
#> Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
#> Chain 1: 
#> Chain 1: Gradient evaluation took 0.000376 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 3.76 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
#> Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 1: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 2: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 1: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 2: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 1: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 7.712 seconds (Warm-up)
#> Chain 1:                23.864 seconds (Sampling)
#> Chain 1:                31.576 seconds (Total)
#> Chain 1: 
#> Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 8.315 seconds (Warm-up)
#> Chain 2:                40.132 seconds (Sampling)
#> Chain 2:                48.447 seconds (Total)
#> Chain 2:
```

### Model Evaluation

``` r

BH_pred <- predict(model_reg, newdata = BH_test)
measure_cont(BH_test$medv, BH_pred)
#> $rmse
#> [1] 7.869471
#> 
#> $mae
#> [1] 5.282951
```

------------------------------------------------------------------------

## Binary Classification: PimaIndiansDiabetes Dataset

### Dataset Description

The `PimaIndiansDiabetes` dataset contains features related to health
status for predicting the presence of diabetes.

``` r

data(PimaIndiansDiabetes)
PID_data <- PimaIndiansDiabetes |>
  transform(diabetes = ifelse(diabetes == "pos", 1, 0))
# Splitting data into training and testing sets
PID_split <- initial_split(PID_data, prop = 0.8, strata = "diabetes")
PID_train <- training(PID_split)
PID_test <- testing(PID_split)
```

### Model Training

``` r

model_bin <- bnns(
  diabetes ~ -1 + .,
  data = PID_train, L = 2,
  out_act_fn = 2, iter = 1e3, warmup = 2e2, chains = 2, cores = 2
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
#> Chain 1: Gradient evaluation took 0.000361 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 3.61 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
#> Chain 2: 
#> Chain 2: Gradient evaluation took 0.000358 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 3.58 seconds.
#> Chain 2: Adjust your expectations accordingly!
#> Chain 2: 
#> Chain 2: 
#> Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
#> Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
#> Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 1: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 1: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 1: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
#> Chain 2: Iteration: 201 / 1000 [ 20%]  (Sampling)
#> Chain 1: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 8.474 seconds (Warm-up)
#> Chain 1:                27.532 seconds (Sampling)
#> Chain 1:                36.006 seconds (Total)
#> Chain 1: 
#> Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 13.173 seconds (Warm-up)
#> Chain 2:                52.74 seconds (Sampling)
#> Chain 2:                65.913 seconds (Total)
#> Chain 2:
```

### Model Evaluation

``` r

PID_pred <- predict(model_bin, newdata = PID_test)
PID_measure <- measure_bin(PID_test$diabetes, PID_pred)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
PID_measure
#> $conf_mat
#>    pred_label
#> obs  0  1
#>   0 87 13
#>   1 27 27
#> 
#> $accuracy
#> [1] 0.7402597
#> 
#> $ROC
#> 
#> Call:
#> roc.default(response = obs, predictor = pred)
#> 
#> Data: pred in 100 controls (obs 0) < 54 cases (obs 1).
#> Area under the curve: 0.7702
#> 
#> $AUC
#> [1] 0.7701852
plot(PID_measure$ROC)
```

![](common_applications_files/figure-html/binary-evaluation-1.png)

------------------------------------------------------------------------

## Multi-class Classification: Glass Dataset

### Dataset Description

The `Glass` dataset contains features to classify glass types.

``` r

data(Glass)
Glass_data <- Glass

# Splitting data into training and testing sets
Glass_split <- initial_split(Glass_data, prop = 0.8, strata = "Type")
Glass_train <- training(Glass_split)
Glass_test <- testing(Glass_split)
```

### Model Training

``` r

model_multi <- bnns(
  Type ~ -1 + .,
  data = Glass_train, L = 2,
  out_act_fn = 3, iter = 1e3, warmup = 2e2, chains = 2, cores = 2
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
#> Chain 1: Gradient evaluation took 0.000282 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 2.82 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
#> 
#> SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
#> Chain 2: 
#> Chain 2: Gradient evaluation took 0.000284 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 2.84 seconds.
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
#> Chain 1: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 2: Iteration: 300 / 1000 [ 30%]  (Sampling)
#> Chain 1: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 400 / 1000 [ 40%]  (Sampling)
#> Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 2: Iteration: 500 / 1000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
#> Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 9.03 seconds (Warm-up)
#> Chain 1:                30.026 seconds (Sampling)
#> Chain 1:                39.056 seconds (Total)
#> Chain 1: 
#> Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 8.839 seconds (Warm-up)
#> Chain 2:                43.325 seconds (Sampling)
#> Chain 2:                52.164 seconds (Total)
#> Chain 2:
```

### Model Evaluation

``` r

Glass_pred <- predict(model_multi, newdata = Glass_test)
measure_cat(Glass_test$Type, Glass_pred)
#> $log_loss
#> [1] 1.226032
#> 
#> $ROC
#> 
#> Call:
#> multiclass.roc.default(response = obs, predictor = `colnames<-`(data.frame(pred),     levels(obs)))
#> 
#> Data: multivariate predictor `colnames<-`(data.frame(pred), levels(obs)) with 6 levels of obs: 1, 2, 3, 5, 6, 7.
#> Multi-class area under the curve: 0.7892
#> 
#> $AUC
#> [1] 0.7891865
```

------------------------------------------------------------------------

## Summary

The performance of the `bnns` package demonstrates its flexibility
across various machine learning tasks. It provides posterior
distributions of predictions, which can be used for uncertainty
quantification and probabilistic decision-making.
