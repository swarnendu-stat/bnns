# bnns 0.1.3

* **New Feature**: Added full support for the `tidymodels` ecosystem, including integration with `parsnip`, `recipes`, `workflows`, and `tune`.
* **New Feature**: Added GPU acceleration support for fitting Bayesian Neural Networks via OpenCL and `cmdstanr`.
* **New Feature**: Introduced model evaluation metrics using Leave-One-Out (LOO) cross-validation (`loo.bnns()`) and Watanabe-Akaike Information Criterion (`waic.bnns()`).
* **New Feature**: Added `save_bnns()` and `load_bnns()` functions for easy saving and loading of trained models.
* **New Feature**: Added a `plot.bnns()` method for visualizing posterior distributions and model convergence.
* Changed the package to stable lifecycle and prepared for CRAN submission.
* Added `Additional_repositories: https://mc-stan.org/r-packages/` to fix `cmdstanr` package resolution.
* Removed the manual rhub GitHub Actions badge from the README.
* Added `CODE_OF_CONDUCT.md`.

# bnns 0.1.2.9000

* Added a `NEWS.md` file to track changes to the package.
