## bnns 1.0.0

This is an update to an existing package. Key changes in this release include:
* Enhanced the `act_fn` and `out_act_fn` arguments to support character string inputs for a better user experience.
* `tidymodels` integration.
* GPU support with OpenCL (with `cmdstanr` accessed dynamically per CRAN policies).

## Resubmission

This is a resubmission to address the Windows incoming pre-test NOTE:

* Overall checktime 15 min > 10 min

To reduce check time, remaining expensive Stan-sampling tests are now skipped
on CRAN, computational vignette chunks are shown but not evaluated, and
pkgdown-only article sources are excluded from the CRAN source tarball.

## R CMD check results

0 errors | 0 warnings | 0 notes
