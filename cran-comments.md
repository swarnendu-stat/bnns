## bnns 1.0.0

This is an update to an existing package. Key changes in this release include:
* Enhanced the `act_fn` and `out_act_fn` arguments to support character string inputs for a better user experience.
* `tidymodels` integration
* GPU support with OpenCL

## R CMD check results

0 errors ✔ | 0 warnings ✔ | 1 note ✖

* The note regarding 'Remotes' and 'cmdstanr' not being in mainstream repositories is expected. The `cmdstanr` package is not on CRAN (it is hosted by the Stan Development Team). It is only listed in `Suggests` to provide users with an alternative faster backend and GPU support. The package degrades gracefully and functions perfectly with the default `rstan` backend (which is on CRAN) if `cmdstanr` is not installed. The `Remotes` field is included to facilitate automated installation of `cmdstanr` for users.
