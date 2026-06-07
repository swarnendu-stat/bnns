## bnns 1.0.0

This is an update to an existing package. Key changes in this release include:
* Enhanced the `act_fn` and `out_act_fn` arguments to support character string inputs for a better user experience.
* `tidymodels` integration
* GPU support with OpenCL

## Re-Submission to reduce test-time

`skip_on_cran()` has been added to lengthy integration tests to resolve the Windows check time limit.

## R CMD check results

0 errors ✔ | 0 warnings ✔ | 0 notes ✔
