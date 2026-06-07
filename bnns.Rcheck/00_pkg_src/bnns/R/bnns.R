#' Generic Function for Fitting Bayesian Neural Network Models
#'
#' This is a generic function for fitting Bayesian Neural Network (BNN) models. It dispatches to methods based on the class of the input data.
#'
#' @param formula A symbolic description of the model to be fitted. The formula should specify the response variable and predictors (e.g., \code{y ~ x1 + x2}). \code{y} must be continuous for regression (`out_act_fn = 1`), numeric 0/1 for binary classification (`out_act_fn = 2`), and factor with at least 3 levels for multi-classification (`out_act_fn = 3`).
#' @param data A data frame containing the variables in the model.
#' @param L An integer specifying the number of hidden layers in the neural network. Default is 1.
#' @param nodes An integer or vector specifying the number of nodes in each hidden layer. If a single value is provided, it is applied to all layers. Default is 16.
#' @param act_fn An integer or vector specifying the activation function(s) for the hidden layers. Options are:
#'   \itemize{
#'     \item \code{1} for tanh
#'     \item \code{2} for sigmoid (default)
#'     \item \code{3} for softplus
#'     \item \code{4} for ReLU
#'     \item \code{5} for linear
#'   }
#' @param out_act_fn An integer or character string specifying the activation function for the output layer. Options are:
#'   \itemize{
#'     \item \code{1} or \code{"linear"} for linear (default)
#'     \item \code{2} or \code{"sigmoid"} for sigmoid
#'     \item \code{3} or \code{"softmax"} for softmax
#'   }
#' @param algorithm A character string specifying the MCMC algorithm. Options are \code{"NUTS"} (default) or \code{"HMC"}.
#' @param iter An integer specifying the total number of iterations for the Stan sampler. Default is \code{1e3}.
#' @param warmup An integer specifying the number of warmup iterations for the Stan sampler. Default is \code{2e2}.
#' @param thin An integer specifying the thinning interval for Stan samples. Default is 1.
#' @param chains An integer specifying the number of Markov chains. Default is 2.
#' @param cores An integer specifying the number of CPU cores to use for parallel sampling. Default is 2.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#' @param prior_weights A list specifying the prior distribution for the weights in the neural network.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"normal"}, \code{"uniform"}, \code{"cauchy"}, and \code{"horseshoe"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"normal"}: Provide \code{mean} (mean of the distribution) and \code{sd} (standard deviation).
#'         \item For \code{"uniform"}: Provide \code{alpha} (lower bound) and \code{beta} (upper bound).
#'         \item For \code{"cauchy"}: Provide \code{mu} (location parameter) and \code{sigma} (scale parameter).
#'       }
#'   }
#'   For the \code{"horseshoe"} prior, `params` is not needed as it uses a standard half-Cauchy setup.
#'   If \code{prior_weights} is \code{NULL}, the default prior is a \code{normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "uniform", params = list(alpha = -1, beta = 1))}
#'     \item \code{list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))}
#'   }
#' @param prior_bias A list specifying the prior distribution for the biases in the neural network.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"normal"}, \code{"uniform"}, \code{"cauchy"}, and \code{"horseshoe"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"normal"}: Provide \code{mean} (mean of the distribution) and \code{sd} (standard deviation).
#'         \item For \code{"uniform"}: Provide \code{alpha} (lower bound) and \code{beta} (upper bound).
#'         \item For \code{"cauchy"}: Provide \code{mu} (location parameter) and \code{sigma} (scale parameter).
#'       }
#'   }
#'   If \code{prior_bias} is \code{NULL}, the default prior is a \code{normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "uniform", params = list(alpha = -1, beta = 1))}
#'     \item \code{list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))}
#'   }
#' @param prior_sigma A list specifying the prior distribution for the \code{sigma} parameter in regression
#'   models (\code{out_act_fn = 1}). This allows for setting priors on the standard deviation of the residuals.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"half-normal"} and \code{"inverse-gamma"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"half-normal"}: Provide \code{sd} (standard deviation of the half-normal distribution).
#'         \item For \code{"inverse-gamma"}: Provide \code{shape} (shape parameter) and \code{scale} (scale parameter).
#'       }
#'   }
#'   If \code{prior_sigma} is \code{NULL}, the default prior is a \code{half-normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "half_normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "inv_gamma", params = list(alpha = 1, beta = 1))}
#'   }
#' @param verbose TRUE or FALSE: flag indicating whether to print intermediate output from Stan on the console, which might be helpful for model debugging.
#' @param refresh refresh (integer) can be used to control how often the progress of the sampling is reported (i.e. show the progress every refresh iterations). By default, refresh = max(iter/10, 1). The progress indicator is turned off if refresh <= 0.
#' @param normalize Logical. If `TRUE` (default), the input predictors
#'   are normalized to have zero mean and unit variance before training.
#'   Normalization ensures stable and efficient Bayesian sampling by standardizing
#'   the input scale, which is particularly beneficial for neural network training.
#'   If `FALSE`, no normalization is applied, and it is assumed that the input data
#'   is already pre-processed appropriately.
#' @param backend A character string specifying the Stan backend to use. Options are \code{"rstan"} (default) or \code{"cmdstanr"}.
#' @param use_gpu Logical. If \code{TRUE}, enables OpenCL for GPU acceleration. Default is \code{FALSE}. (Requires the \code{"cmdstanr"} backend).
#' @param opencl_ids A vector of two integers specifying the OpenCL platform and device IDs. Default is \code{c(0, 0)}.
#' @param ... Currently not in use.
#'
#' @return The result of the method dispatched by the class of the input data. Typically, this would be an object of class \code{"bnns"} containing the fitted model and associated information.
#'
#' @details The function serves as a generic interface to different methods of fitting Bayesian Neural Networks. The specific method dispatched depends on the class of the input arguments, allowing for flexibility in the types of inputs supported.
#'
#' @references
#' 1. Bishop, C.M., 1995. Neural networks for pattern recognition. Oxford university press.
#' 2. Carpenter, B., Gelman, A., Hoffman, M.D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M.A., Guo, J., Li, P. and Riddell, A., 2017. Stan: A probabilistic programming language. Journal of statistical software, 76.
#' 3. Neal, R.M., 2012. Bayesian learning for neural networks (Vol. 118). Springer Science & Business Media.
#'
#' @examples
#' \donttest{
#' # Example usage with formula interface:
#' data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
#' model <- bnns(y ~ -1 + x1 + x2,
#'   data = data, L = 1, nodes = 2, act_fn = 1,
#'   iter = 1e1, warmup = 5, chains = 1
#' )
#' }
#' # See the documentation for bnns.default for more details on the default implementation.
#'
#' @seealso \code{\link{bnns.default}}
#' @importFrom stats median predict
#' @import BH
#' @import RcppEigen
#'
#' @export

bnns <- function(formula, data, L = 1, nodes = rep(2, L),
                 act_fn = rep(2, L), out_act_fn = 1, algorithm = c("NUTS", "HMC"),
                 iter = 1e3, warmup = 2e2,
                 thin = 1, chains = 2, cores = 2, seed = 123, prior_weights = NULL,
                 prior_bias = NULL, prior_sigma = NULL, verbose = FALSE,
                 refresh = max(iter / 10, 1), normalize = TRUE, 
                 backend = c("rstan", "cmdstanr"),
                 use_gpu = FALSE, opencl_ids = c(0, 0), ...) {
  UseMethod("bnns")
}

# Environment to cache compiled Stan models for the duration of the R session
.stan_model_cache <- new.env(parent = emptyenv())

#' Get or compile a Stan model
#' @keywords internal
#' @noRd
get_or_compile_stan_model <- function(stan_code) {
  code_hash <- digest::digest(stan_code, algo = "md5")
  
  if (!is.null(.stan_model_cache[[code_hash]])) {
    return(.stan_model_cache[[code_hash]])
  }
  
  model <- rstan::stan_model(model_code = stan_code)
  .stan_model_cache[[code_hash]] <- model
  return(model)
}

#' Get or compile a CmdStan model
#' @keywords internal
#' @noRd
get_or_compile_cmdstan_model <- function(stan_code, use_gpu = FALSE) {
  hash_input <- list(code = stan_code, use_gpu = use_gpu)
  code_hash <- digest::digest(hash_input, algo = "md5")
  
  if (!is.null(.stan_model_cache[[code_hash]])) {
    return(.stan_model_cache[[code_hash]])
  }
  
  cache_dir <- tools::R_user_dir("bnns", which = "cache")
  if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
  
  stan_file <- file.path(cache_dir, paste0("model_", code_hash, ".stan"))
  if (!file.exists(stan_file)) {
    writeLines(stan_code, stan_file)
  }
  
  cpp_options <- list()
  if (use_gpu) {
    cpp_options$stan_opencl <- TRUE
  }

  # cmdstanr automatically reuses the compiled executable if it exists next to the .stan file
  cmdstan_model <- getExportedValue("cmdstanr", "cmdstan_model")
  .stan_model_cache[[code_hash]] <- cmdstan_model(stan_file = stan_file, cpp_options = cpp_options)
  return(.stan_model_cache[[code_hash]])
}

#' Internal function for training the BNN
#'
#' This function performs the actual fitting of the Bayesian Neural Network.
#' It is called by the exported bnns methods and is not intended for direct use.
#'
#' @param train_x A numeric matrix representing the input features (predictors) for training. Rows correspond to observations, and columns correspond to features.
#' @param train_y A numeric vector representing the target values for training. Its length must match the number of rows in `train_x`.
#' @param L An integer specifying the number of hidden layers in the neural network. Default is 1.
#' @param nodes An integer or vector specifying the number of nodes in each hidden layer. If a single value is provided, it is applied to all layers. Default is 16.
#' @param act_fn An integer or vector specifying the activation function(s) for the hidden layers. Options are:
#'   \itemize{
#'     \item \code{1} for tanh
#'     \item \code{2} for sigmoid (default)
#'     \item \code{3} for softplus
#'     \item \code{4} for ReLU
#'     \item \code{5} for linear
#'   }
#' @param out_act_fn An integer or character string specifying the activation function for the output layer. Options are:
#'   \itemize{
#'     \item \code{1} or \code{"linear"} for linear (default)
#'     \item \code{2} or \code{"sigmoid"} for sigmoid
#'     \item \code{3} or \code{"softmax"} for softmax
#'   }
#' @param algorithm A character string specifying the MCMC algorithm. Options are \code{"NUTS"} (default) or \code{"HMC"}.
#' @param iter An integer specifying the total number of iterations for the Stan sampler. Default is \code{1e3}.
#' @param warmup An integer specifying the number of warmup iterations for the Stan sampler. Default is \code{2e2}.
#' @param thin An integer specifying the thinning interval for Stan samples. Default is 1.
#' @param chains An integer specifying the number of Markov chains. Default is 2.
#' @param cores An integer specifying the number of CPU cores to use for parallel sampling. Default is 2.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#' @param prior_weights A list specifying the prior distribution for the weights in the neural network.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"normal"}, \code{"uniform"}, and \code{"cauchy"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"normal"}: Provide \code{mean} (mean of the distribution) and \code{sd} (standard deviation).
#'         \item For \code{"uniform"}: Provide \code{alpha} (lower bound) and \code{beta} (upper bound).
#'         \item For \code{"cauchy"}: Provide \code{mu} (location parameter) and \code{sigma} (scale parameter).
#'       }
#'   }
#'   For the \code{"horseshoe"} prior, `params` is not needed as it uses a standard half-Cauchy setup.
#'   If \code{prior_weights} is \code{NULL}, the default prior is a \code{normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "uniform", params = list(alpha = -1, beta = 1))}
#'     \item \code{list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))}
#'   }
#' @param prior_bias A list specifying the prior distribution for the biases in the neural network.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"normal"}, \code{"uniform"}, and \code{"cauchy"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"normal"}: Provide \code{mean} (mean of the distribution) and \code{sd} (standard deviation).
#'         \item For \code{"uniform"}: Provide \code{alpha} (lower bound) and \code{beta} (upper bound).
#'         \item For \code{"cauchy"}: Provide \code{mu} (location parameter) and \code{sigma} (scale parameter).
#'       }
#'   }
#'   If \code{prior_bias} is \code{NULL}, the default prior is a \code{normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "uniform", params = list(alpha = -1, beta = 1))}
#'     \item \code{list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))}
#'   }
#' @param prior_sigma A list specifying the prior distribution for the \code{sigma} parameter in regression
#'   models (\code{out_act_fn = 1}). This allows for setting priors on the standard deviation of the residuals.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"half-normal"} and \code{"inverse-gamma"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"half-normal"}: Provide \code{sd} (standard deviation of the half-normal distribution).
#'         \item For \code{"inverse-gamma"}: Provide \code{shape} (shape parameter) and \code{scale} (scale parameter).
#'       }
#'   }
#'   If \code{prior_sigma} is \code{NULL}, the default prior is a \code{half-normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "half_normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "inv_gamma", params = list(alpha = 1, beta = 1))}
#'   }
#' @param verbose TRUE or FALSE: flag indicating whether to print intermediate output from Stan on the console, which might be helpful for model debugging.
#' @param refresh refresh (integer) can be used to control how often the progress of the sampling is reported (i.e. show the progress every refresh iterations). By default, refresh = max(iter/10, 1). The progress indicator is turned off if refresh <= 0.
#' @param normalize Logical. If `TRUE` (default), the input predictors
#'   are normalized to have zero mean and unit variance before training.
#'   Normalization ensures stable and efficient Bayesian sampling by standardizing
#'   the input scale, which is particularly beneficial for neural network training.
#'   If `FALSE`, no normalization is applied, and it is assumed that the input data
#'   is already pre-processed appropriately.
#' @param backend A character string specifying the Stan backend to use. Options are \code{"rstan"} (default) or \code{"cmdstanr"}.
#' @param use_gpu Logical. If \code{TRUE}, enables OpenCL for GPU acceleration. Default is \code{FALSE}. (Requires the \code{"cmdstanr"} backend).
#' @param opencl_ids A vector of two integers specifying the OpenCL platform and device IDs. Default is \code{c(0, 0)}.
#' @param ... Currently not in use.
#'
#' @return An object of class \code{"bnns"} containing the following components:
#' \describe{
#'   \item{\code{fit}}{The fitted Stan model object.}
#'   \item{\code{call}}{The matched call.}
#'   \item{\code{data}}{A list containing the Stan data used in the model.}
#' }
#'
#' @details The function uses the \code{generate_stan_code} function to dynamically generate Stan code based on the specified number of layers and nodes. Stan is then used to fit the Bayesian Neural Network.
#'
#' @examples
#' \donttest{
#' # Example usage:
#' train_x <- matrix(runif(20), nrow = 10, ncol = 2)
#' train_y <- rnorm(10)
#' model <- bnns::bnns_train(train_x, train_y,
#'   L = 1, nodes = 2, act_fn = 2,
#'   iter = 1e1, warmup = 5, chains = 1
#' )
#'
#' # Access Stan model fit
#' model$fit
#' }
#' @seealso \code{\link[rstan]{stan}}
#'
#' @export
#' @keywords internal

bnns_train <- function(train_x,
                       train_y,
                       L = 1,
                       nodes = rep(2, L),
                       act_fn = rep(2, L),
                       out_act_fn = 1,
                       algorithm = c("NUTS", "HMC"),
                       iter = 1000,
                       warmup = 200,
                       thin = 1,
                       chains = 2,
                       cores = 2,
                       seed = 123,
                       prior_weights = NULL,
                       prior_bias = NULL,
                       prior_sigma = NULL,
                       verbose = FALSE,
                       refresh = max(iter/10, 1),
                       normalize = TRUE,
                       backend = c("rstan", "cmdstanr"),
                       use_gpu = FALSE,
                       opencl_ids = c(0, 0),
                       ...) {
  
  backend <- match.arg(backend)
  algorithm <- match.arg(toupper(algorithm[1]), c("NUTS", "HMC"))
  
  if (use_gpu && backend != "cmdstanr") {
    warning("GPU acceleration is only supported with the 'cmdstanr' backend. Switching backend to 'cmdstanr'.", call. = FALSE)
    backend <- "cmdstanr"
  }

  if (backend == "cmdstanr" && algorithm == "HMC") {
    warning("The 'cmdstanr' backend does not natively expose static HMC via the sample method. Using the default NUTS engine instead.", call. = FALSE)
  }

  if (use_gpu && backend == "cmdstanr" && requireNamespace("OpenCL", quietly = TRUE)) {
    has_opencl <- tryCatch(length(OpenCL::oclPlatforms()) > 0, error = function(e) FALSE)
    if (!has_opencl) {
      warning("No OpenCL platforms found on this system. Falling back to CPU.", call. = FALSE)
      use_gpu <- FALSE
    }
  }
  if (missing(train_x)) stop("Argument train_x is missing")
  if (missing(train_y)) stop("Argument train_y is missing")
  if (!((L %% 1 == 0) & (sign(L) == 1))) stop("L must be a positive integer")
  
  if (!(all(nodes %% 1 == 0) & all(sign(nodes) == 1))) stop("nodes must be positive integer(s)")
  
  if (is.character(act_fn)) {
    act_fn <- translate_activation(act_fn)
  }
  if (is.character(out_act_fn)) {
    out_act_fn <- translate_out_activation(out_act_fn)
  }

  if (!all(act_fn %in% 1:5)) stop("act_fn must be a sequence of 1/2/3/4/5")
  if (!out_act_fn %in% 1:3) stop("out_act_fn must be 1, 2, or 3")

  # Validate that the length of nodes matches the number of layers L
  if (length(nodes) != L) {
    stop("nodes must be of length L", call. = FALSE)
  }
  
  # Validate that the length of act_fn matches the number of layers L
  if (length(act_fn) != L) {
    stop("act_fn must be of length L", call. = FALSE)
  }
  

  # Default priors: Normal(0, 1)
  if (is.null(prior_weights)) {
    prior_weights <- list(
      dist = "normal",
      params = list(mean = 0, sd = 1)
    )
  }

  # Validate the prior specification
  if (!is.list(prior_weights) || !("dist" %in% names(prior_weights)) || !("params" %in% names(prior_weights))) {
    stop("'prior_weights' must be a list with elements 'dist' and 'params'.")
  }

  supported_distributions <- c("normal", "uniform", "cauchy", "horseshoe")
  if (!(prior_weights$dist %in% supported_distributions)) {
    stop(paste(
      "Unsupported distribution for weights:", prior_weights$dist,
      ". Supported distributions are:", paste(supported_distributions, collapse = ", ")
    ))
  }

  # Default priors: Normal(0, 1)
  if (is.null(prior_bias)) {
    prior_bias <- list(
      dist = "normal",
      params = list(mean = 0, sd = 1)
    )
  }

  # Validate the prior specification
  if (!is.list(prior_bias) || !("dist" %in% names(prior_bias)) || !("params" %in% names(prior_bias))) {
    stop("'prior_bias' must be a list with elements 'dist' and 'params'.")
  }

  supported_distributions <- c("normal", "uniform", "cauchy")
  if (!(prior_bias$dist %in% supported_distributions)) {
    stop(paste(
      "Unsupported distribution for biases:", prior_bias$dist,
      ". Supported distributions are:", paste(supported_distributions, collapse = ", ")
    ))
  }

  # Validate parameters for the chosen distribution
  validate_prior <- function(dist, params) {
    if (dist == "horseshoe") {
      return() # No params to validate for horseshoe
    }
    switch(dist,
      normal = {
        if (!all(c("mean", "sd") %in% names(params))) {
          stop("For 'normal' distribution, 'params' must contain 'mean' and 'sd'.")
        }
      },
      uniform = {
        if (!all(c("alpha", "beta") %in% names(params))) {
          stop("For 'uniform' distribution, 'params' must contain 'alpha' and 'beta'.")
        }
      },
      cauchy = {
        if (!all(c("mu", "sigma") %in% names(params))) {
          stop("For 'cauchy' distribution, 'params' must contain 'mu' and 'sigma'.")
        }
      },
      half_normal = {
        if (!all(c("mean", "sd") %in% names(params))) {
          stop("For 'half_normal' distribution, 'params' must contain 'mean' and 'sd'.")
        }
      },
      inv_gamma = {
        if (!all(c("alpha", "beta") %in% names(params))) {
          stop("For 'inv_gamma' distribution, 'params' must contain 'alpha' and 'beta'.")
        }
      }
    )
  }

  validate_prior(prior_weights$dist, prior_weights$params)
  validate_prior(prior_bias$dist, prior_bias$params)

  prior_specification_weights <- ""
  if (prior_weights$dist != "horseshoe") {
    # Replace PRIOR_SPECIFICATION with the appropriate Stan syntax
    prior_specification_weights <- switch(prior_weights$dist,
      normal = sprintf("normal(%f, %f)", prior_weights$params$mean, prior_weights$params$sd),
      uniform = sprintf("uniform(%f, %f)", prior_weights$params$alpha, prior_weights$params$beta),
      cauchy = sprintf("cauchy(%f, %f)", prior_weights$params$mu, prior_weights$params$sigma)
    )
  }

  prior_specification_bias <- switch(prior_bias$dist,
    normal = sprintf("normal(%f, %f)", prior_bias$params$mean, prior_bias$params$sd),
    uniform = sprintf("uniform(%f, %f)", prior_bias$params$alpha, prior_bias$params$beta),
    cauchy = sprintf("cauchy(%f, %f)", prior_bias$params$mu, prior_bias$params$sigma)
  )

  if (normalize) {
    x_mean <- colMeans(train_x)
    x_sd <- apply(train_x, 2, stats::sd)
    x_sd[which(x_sd == 0)] <- 1
    train_x <- sweep(train_x, 2, x_mean, "-") / x_sd
  }

  if (out_act_fn == 3) {
    stopifnot("train_y must be a factor" = is.factor(train_y))
    stopifnot("train_y must have at least 3 levels" = length(levels(train_y)) >= 3)
    stan_data <- list(
      n = nrow(train_x), # Number of observations
      m = ncol(train_x), # Number of features
      L = L,
      nodes = as.array(nodes), # Number of layers
      X = train_x, # Input matrix
      y = as.numeric(train_y), # Output vector
      K = length(unique(train_y)),
      act_fn = as.array(act_fn), # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  } else if (out_act_fn == 2) {
    stopifnot("train_y must have only 0/1 values" = all(train_y %in% c(0, 1)))
    stan_data <- list(
      n = nrow(train_x), # Number of observations
      m = ncol(train_x), # Number of features
      L = L,
      nodes = as.array(nodes), # Number of layers
      X = train_x, # Input matrix
      y = train_y, # Output vector
      act_fn = as.array(act_fn), # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  } else {
    stan_data <- list(
      n = nrow(train_x), # Number of observations
      m = ncol(train_x), # Number of features
      L = L,
      nodes = as.array(nodes), # Number of layers
      X = train_x, # Input matrix
      y = train_y, # Output vector
      act_fn = as.array(act_fn), # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  }

  est <- list()
  pars <- c(paste0("w", 1:L), paste0("b", 1:L), "w_out", "b_out", "log_lik", "y_rep")
  if (out_act_fn == 1) {
    pars <- c(pars, "sigma")
  }

  stan_model <- generate_stan_code(num_layers = L, nodes = nodes, out_act_fn = out_act_fn, prior_weights_dist = prior_weights$dist)
  if (prior_weights$dist != "horseshoe") {
    stan_model <- gsub("PRIOR_WEIGHT", prior_specification_weights, stan_model)
  }
  stan_model <- gsub(x = stan_model, pattern = "PRIOR_BIAS", replacement = prior_specification_bias)

  # Check prior_sigma (only relevant for regression models)
  if (out_act_fn == 1) {
    if (!is.null(prior_sigma)) {
      if (!all(c("dist", "params") %in% names(prior_sigma))) {
        stop("'prior_sigma' must contain 'dist' and 'params' elements.")
      }
      if (!prior_sigma$dist %in% c("half_normal", "inv_gamma")) {
        stop("Supported prior distributions for sigma are 'half_normal' and 'inv_gamma'.")
      }
    } else {
      # Default prior for sigma
      prior_sigma <- list(dist = "half_normal", params = list(mean = 0, sd = 1))
    }

    validate_prior(prior_sigma$dist, prior_sigma$params)

    # Replace PRIOR_SIGMA with the appropriate Stan syntax
    prior_sigma <- switch(prior_sigma$dist,
      half_normal = sprintf("normal(%f, %f)", prior_sigma$params$mean, prior_sigma$params$sd),
      inv_gamma = sprintf("inv_gamma(%f, %f)", prior_sigma$params$alpha, prior_sigma$params$beta)
    )

    stan_model <- gsub(x = stan_model, pattern = "PRIOR_SIGMA", replacement = prior_sigma)
  }
  
  if (backend == "cmdstanr") {
    if (!has_namespace("cmdstanr")) {
      stop("The 'cmdstanr' package is required for the cmdstanr backend. Install it using: install.packages('cmdstanr', repos = c('https://mc-stan.org/r-packages/', getOption('repos')))", call. = FALSE)
    }
    
    compiled_model <- NULL
    if (use_gpu) {
      tryCatch({
        compiled_model <- get_or_compile_cmdstan_model(stan_model, TRUE)
      }, error = function(e) {
        warning("GPU compilation failed. Falling back to CPU. Reason: ", conditionMessage(e), call. = FALSE)
      })
    }
    
    if (is.null(compiled_model)) {
      use_gpu <- FALSE
      compiled_model <- get_or_compile_cmdstan_model(stan_model, FALSE)
    }
    
    sample_args <- list(
      data = stan_data,
      seed = seed,
      chains = chains,
      parallel_chains = cores,
      iter_warmup = warmup,
      iter_sampling = iter - warmup,
      thin = thin,
      refresh = refresh,
      show_messages = verbose
    )
    
    if (use_gpu) {
      sample_args$opencl_ids <- opencl_ids
      cmd_fit <- tryCatch({
        do.call(compiled_model$sample, sample_args)
      }, error = function(e) {
        warning("GPU sampling failed. Falling back to CPU. Reason: ", conditionMessage(e), call. = FALSE)
        sample_args$opencl_ids <- NULL
        compiled_model_cpu <- get_or_compile_cmdstan_model(stan_model, FALSE)
        do.call(compiled_model_cpu$sample, sample_args)
      })
    } else {
      cmd_fit <- do.call(compiled_model$sample, sample_args)
    }
    
    est$fit <- rstan::read_stan_csv(cmd_fit$output_files())
  } else {
    compiled_model <- get_or_compile_stan_model(stan_model)
    est$fit <- suppressWarnings(rstan::sampling(
      object = compiled_model, data = stan_data, include = TRUE,
      pars = pars, algorithm = algorithm,
      iter = iter, warmup = warmup, thin = thin, chains = chains, cores = cores,
      init = 0, seed = seed, verbose = verbose, refresh = refresh
    ))
  }

  est$call <- match.call()
  est$data <- stan_data
  est$normalize <- normalize
  if (normalize) {
    est$x_mean <- x_mean
    est$x_sd <- x_sd
  }

  class(est) <- "bnns"
  return(est)
}

#' Bayesian Neural Network Model Using Formula(default) Interface
#'
#' Fits a Bayesian Neural Network (BNN) model using a formula interface. The function parses the formula and data to create the input feature matrix and target vector, then fits the model using \code{\link{bnns.default}}.
#'
#' @param formula A symbolic description of the model to be fitted. The formula should specify the response variable and predictors (e.g., \code{y ~ x1 + x2}).
#' @param data A data frame containing the variables in the model.
#' @param L An integer specifying the number of hidden layers in the neural network. Default is 1.
#' @param nodes An integer or vector specifying the number of nodes in each hidden layer. If a single value is provided, it is applied to all layers. Default is 16.
#' @param act_fn An integer or vector specifying the activation function(s) for the hidden layers. Options are:
#'   \itemize{
#'     \item \code{1} for tanh
#'     \item \code{2} for sigmoid (default)
#'     \item \code{3} for softplus
#'     \item \code{4} for ReLU
#'     \item \code{5} for linear
#'   }
#' @param out_act_fn An integer or character string specifying the activation function for the output layer. Options are:
#'   \itemize{
#'     \item \code{1} or \code{"linear"} for linear (default)
#'     \item \code{2} or \code{"sigmoid"} for sigmoid
#'     \item \code{3} or \code{"softmax"} for softmax
#'   }
#' @param algorithm A character string specifying the MCMC algorithm. Options are \code{"NUTS"} (default) or \code{"HMC"}.
#' @param iter An integer specifying the total number of iterations for the Stan sampler. Default is \code{1e3}.
#' @param warmup An integer specifying the number of warmup iterations for the Stan sampler. Default is \code{2e2}.
#' @param thin An integer specifying the thinning interval for Stan samples. Default is 1.
#' @param chains An integer specifying the number of Markov chains. Default is 2.
#' @param cores An integer specifying the number of CPU cores to use for parallel sampling. Default is 2.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#' @param prior_weights A list specifying the prior distribution for the weights in the neural network.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"normal"}, \code{"uniform"}, \code{"cauchy"}, and \code{"horseshoe"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"normal"}: Provide \code{mean} (mean of the distribution) and \code{sd} (standard deviation).
#'         \item For \code{"uniform"}: Provide \code{alpha} (lower bound) and \code{beta} (upper bound).
#'         \item For \code{"cauchy"}: Provide \code{mu} (location parameter) and \code{sigma} (scale parameter).
#'       }
#'   }
#'   For the \code{"horseshoe"} prior, `params` is not needed as it uses a standard half-Cauchy setup.
#'   If \code{prior_weights} is \code{NULL}, the default prior is a \code{normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "uniform", params = list(alpha = -1, beta = 1))}
#'     \item \code{list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))}
#'   }
#' @param prior_bias A list specifying the prior distribution for the biases in the neural network.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"normal"}, \code{"uniform"}, and \code{"cauchy"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"normal"}: Provide \code{mean} (mean of the distribution) and \code{sd} (standard deviation).
#'         \item For \code{"uniform"}: Provide \code{alpha} (lower bound) and \code{beta} (upper bound).
#'         \item For \code{"cauchy"}: Provide \code{mu} (location parameter) and \code{sigma} (scale parameter).
#'       }
#'   }
#'   If \code{prior_bias} is \code{NULL}, the default prior is a \code{normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "uniform", params = list(alpha = -1, beta = 1))}
#'     \item \code{list(dist = "cauchy", params = list(mu = 0, sigma = 2.5))}
#'   }
#' @param prior_sigma A list specifying the prior distribution for the \code{sigma} parameter in regression
#'   models (\code{out_act_fn = 1}). This allows for setting priors on the standard deviation of the residuals.
#'   The list must include two components:
#'   \itemize{
#'     \item \code{dist}: A character string specifying the distribution type. Supported values are
#'       \code{"half-normal"} and \code{"inverse-gamma"}.
#'     \item \code{params}: A named list specifying the parameters for the chosen distribution:
#'       \itemize{
#'         \item For \code{"half-normal"}: Provide \code{sd} (standard deviation of the half-normal distribution).
#'         \item For \code{"inverse-gamma"}: Provide \code{shape} (shape parameter) and \code{scale} (scale parameter).
#'       }
#'   }
#'   If \code{prior_sigma} is \code{NULL}, the default prior is a \code{half-normal(0, 1)} distribution.
#'   For example:
#'   \itemize{
#'     \item \code{list(dist = "half_normal", params = list(mean = 0, sd = 1))}
#'     \item \code{list(dist = "inv_gamma", params = list(alpha = 1, beta = 1))}
#'   }
#' @param verbose TRUE or FALSE: flag indicating whether to print intermediate output from Stan on the console, which might be helpful for model debugging.
#' @param refresh refresh (integer) can be used to control how often the progress of the sampling is reported (i.e. show the progress every refresh iterations). By default, refresh = max(iter/10, 1). The progress indicator is turned off if refresh <= 0.
#' @param normalize Logical. If `TRUE` (default), the input predictors
#'   are normalized to have zero mean and unit variance before training.
#'   Normalization ensures stable and efficient Bayesian sampling by standardizing
#'   the input scale, which is particularly beneficial for neural network training.
#'   If `FALSE`, no normalization is applied, and it is assumed that the input data
#'   is already pre-processed appropriately.
#' @param backend A character string specifying the Stan backend to use. Options are \code{"rstan"} (default) or \code{"cmdstanr"}.
#' @param use_gpu Logical. If \code{TRUE}, enables OpenCL for GPU acceleration. Default is \code{FALSE}. (Requires the \code{"cmdstanr"} backend).
#' @param opencl_ids A vector of two integers specifying the OpenCL platform and device IDs. Default is \code{c(0, 0)}.
#' @param ... Currently not in use.
#'
#' @return An object of class \code{"bnns"} containing the fitted model and associated information, including:
#'   \itemize{
#'     \item \code{fit}: The fitted Stan model object.
#'     \item \code{data}: A list containing the processed training data.
#'     \item \code{call}: The matched function call.
#'     \item \code{formula}: The formula used for the model.
#'   }
#'
#' @details The function uses the provided formula and data to generate the design matrix for the predictors and the response vector. It then calls helper function bnns_train to fit the Bayesian Neural Network model.
#'
#' @examples
#' \donttest{
#' # Example usage:
#' data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
#' model <- bnns(y ~ -1 + x1 + x2,
#'   data = data, L = 1, nodes = 2, act_fn = 3,
#'   iter = 1e1, warmup = 5, chains = 1
#' )
#' }
#' @export

bnns.default <- function(formula, data, L = 1, nodes = rep(2, L),
                         act_fn = rep(2, L), out_act_fn = 1, algorithm = c("NUTS", "HMC"),
                         iter = 1e3, warmup = 2e2,
                         thin = 1, chains = 2, cores = 2, seed = 123, prior_weights = NULL,
                         prior_bias = NULL, prior_sigma = NULL, verbose = FALSE,
                         refresh = max(iter / 10, 1), normalize = TRUE, 
                         backend = c("rstan", "cmdstanr"),
                         use_gpu = FALSE, opencl_ids = c(0, 0), ...) {
  if (missing(formula) && missing(data)) {
    return(invisible(NULL))
  }
  if (missing(formula) || missing(data)) {
    stop("Both 'formula' and 'data' must be provided.")
  }

  if(!is.data.frame(data)){
    stop("'data' must be a data.frame.")
  }

  if (anyNA(data)) {
    stop("'data' contains missing values. Please handle them before proceeding.")
  }

  if (any(unlist(lapply(data, is.nan))) | any(unlist(lapply(data, is.infinite)))) {
    stop("'data' contains invalid values (NaN/Inf).")
  }

  mf <- stats::model.frame(formula = formula, data = data)
  train_x <- stats::model.matrix(attr(mf, "terms"), data = mf)
  train_y <- stats::model.response(mf)

  # Translate character activations
  if (is.character(act_fn)) {
    act_fn <- translate_activation(act_fn)
  }
  if (is.character(out_act_fn)) {
    out_act_fn <- translate_out_activation(out_act_fn)
  }

  if (length(nodes) == 1 && L > 1) nodes <- rep(nodes, L)
  if (length(act_fn) == 1 && L > 1) act_fn <- rep(act_fn, L)

  levels_y <- NULL
  if (is.factor(train_y)) {
    levels_y <- levels(train_y)
    
    # Auto-correct out_act_fn if it's default 1 but y is a factor
    if (out_act_fn == 1) {
      out_act_fn <- if (length(levels_y) == 2) 2 else 3
    }
    
    # Automatically convert binary factors to 0/1 to prevent bnns_train errors
    if (out_act_fn == 2) {
      train_y <- as.numeric(train_y) - 1
    }
  } else if (!is.numeric(train_y)) {
    stop("The response variable must be a numeric vector or a factor.")
  }
  est <- bnns_train(
    train_x = train_x, train_y = train_y, L = L, nodes = nodes,
    act_fn = act_fn, out_act_fn = out_act_fn, algorithm = algorithm, iter = iter,
    warmup = warmup, thin = thin, chains = chains,
    cores = cores, seed = seed, prior_weights = prior_weights,
    prior_bias = prior_bias, prior_sigma = prior_sigma,
    verbose = verbose, refresh = refresh, normalize = normalize, 
    backend = backend, use_gpu = use_gpu, opencl_ids = opencl_ids, ...
  )
  est$call <- match.call()
  est$formula <- formula
  est$terms <- attr(mf, "terms")
  est$levels <- levels_y
  return(est)
}


#' Leave-One-Out Cross-Validation (LOO) for bnns models
#'
#' @param x A fitted \code{bnns} model object.
#' @param ... Additional arguments passed to \code{loo::loo()}.
#' @return A \code{loo} object containing model comparison metrics.
#' @export
#' @importFrom loo loo
loo.bnns <- function(x, ...) {
  # The loo package directly supports stanfit objects if 'log_lik' is present
  loo::loo(x$fit, ...)
}

#' Watanabe-Akaike Information Criterion (WAIC) for bnns models
#'
#' @param x A fitted \code{bnns} model object.
#' @param ... Additional arguments passed to \code{loo::waic()}.
#' @return A \code{waic} object containing model comparison metrics.
#' @export
#' @importFrom loo waic
waic.bnns <- function(x, ...) {
  # Extract the pointwise log-likelihood matrix from the stanfit object
  # and pass it to loo::waic()
  log_lik_mat <- loo::extract_log_lik(x$fit, merge_chains = FALSE)
  loo::waic(log_lik_mat, ...)
}

#' Plot diagnostics for a fitted Bayesian Neural Network
#'
#' Generates Markov Chain Monte Carlo (MCMC) trace plots, posterior density plots,
#' Posterior Predictive Checks (PPC), or predicted probability distributions for the fitted model.
#'
#' @param x A fitted \code{bnns} model object.
#' @param type Character string indicating the type of plot. 
#'   Options are \code{"trace"} for MCMC trace plots, \code{"density"} for posterior density plots,
#'   \code{"posterior_predictive"} for Posterior Predictive Checks, and \code{"pred_prob"} for
#'   visualizing the predicted class probability distributions (classification only).
#' @param pars A character vector of parameter names to include in the plot. 
#'   By default, this focuses on the output layer (\code{"w_out"}, \code{"b_out"}, and \code{"sigma"}) 
#'   to avoid cluttering the plot device with hundreds of hidden layer weights.
#' @param ... Additional arguments passed to \code{\link[rstan]{stan_trace}}, \code{\link[rstan]{stan_dens}}, 
#'   or \code{\link[bayesplot]{ppc_dens_overlay}}.
#'
#' @return A \code{ggplot} object containing the requested diagnostic plots.
#' @export
plot.bnns <- function(x, type = c("trace", "density", "posterior_predictive", "pred_prob"), pars = NULL, ...) {
  type <- match.arg(type)
  
  if (type == "pred_prob") {
    if (x$data$out_act_fn == 1) {
      stop("type = 'pred_prob' is only applicable for classification models.", call. = FALSE)
    }
    
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
      stop("The 'ggplot2' package is required for this plot. Please install it.", call. = FALSE)
    }
    
    # Avoid R CMD check notes for undefined global variables
    Probability <- Class <- True_Class <- Predicted_Class <- NULL
    
    probs <- predict(x, type = "prob")
    y_true <- as.factor(x$data$y)
    if (!is.null(x$levels)) {
      levels(y_true) <- x$levels
    }
    
    if (x$data$out_act_fn == 2) {
      pos_class <- colnames(probs)[2]
      if (is.null(pos_class)) pos_class <- "1"
      
      plot_data <- data.frame(
        Probability = probs[, 2], 
        Class = y_true
      )
      
      p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = Probability, fill = Class)) +
        ggplot2::geom_density(alpha = 0.5) +
        ggplot2::labs(
          title = paste("Predicted Probabilities for Class:", pos_class),
          x = "Posterior Mean Probability",
          y = "Density"
        ) +
        ggplot2::theme_minimal()
      return(p)
    } else {
      pred_classes <- colnames(probs)
      if (is.null(pred_classes)) pred_classes <- paste0("Class_", seq_len(ncol(probs)))
      
      plot_data <- data.frame(
        True_Class = rep(y_true, times = ncol(probs)),
        Predicted_Class = rep(pred_classes, each = nrow(probs)),
        Probability = as.vector(as.matrix(probs))
      )
      
      p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = Probability, fill = Predicted_Class)) +
        ggplot2::geom_density(alpha = 0.5) +
        ggplot2::facet_wrap(~ True_Class, labeller = ggplot2::label_both) +
        ggplot2::labs(
          title = "Predicted Probabilities by True Class",
          x = "Posterior Mean Probability",
          y = "Density",
          fill = "Predicted Class"
        ) +
        ggplot2::theme_minimal()
      return(p)
    }
  }

  if (type == "posterior_predictive") {
    if (!requireNamespace("bayesplot", quietly = TRUE)) {
      stop("The 'bayesplot' package is required for posterior predictive checks. Please install it.", call. = FALSE)
    }
    
    y <- x$data$y
    yrep <- rstan::extract(x$fit, pars = "y_rep")$y_rep
    
    if (is.null(yrep)) {
      stop("y_rep was not found in the Stan model fit. Ensure your Stan code generates 'y_rep'.", call. = FALSE)
    }
    
    if (x$data$out_act_fn == 1) {
      return(bayesplot::ppc_dens_overlay(y, yrep, ...))
    } else {
      return(bayesplot::ppc_bars(y, yrep, ...))
    }
  }
  
  # Smart defaults to avoid plotting thousands of parameters
  if (is.null(pars)) {
    pars <- c("w_out", "b_out")
    if (x$data$out_act_fn == 1) {
      pars <- c(pars, "sigma")
    }
  }
  
  # Route to rstan plotting functions
  if (type == "trace") {
    return(rstan::stan_trace(x$fit, pars = pars, ...))
  } else {
    return(rstan::stan_dens(x$fit, pars = pars, ...))
  }
}

#' @importFrom tibble as_tibble
#' @keywords internal
predict_prob_bnns <- function(object, new_data) {
  # object is the parsnip fit object, object$fit is the bnns object
  preds <- predict(object$fit, newdata = new_data)

  out_act_fn <- object$fit$data$out_act_fn

  # Multiclass Classification (out_act_fn = 3)
  if (out_act_fn == 3) {
    if (length(dim(preds)) != 3) stop("Expected a 3D array for multiclass predictions.", call. = FALSE)

    # Average across the posterior samples (dimension 2)
    prob_matrix <- apply(preds, c(1, 3), mean)
    colnames(prob_matrix) <- object$lvl # Use levels from parsnip object
    return(tibble::as_tibble(prob_matrix))
  }

  # Binary Classification (out_act_fn = 2)
  if (out_act_fn == 2) {
    if (length(dim(preds)) != 2) stop("Expected a 2D array for binary classification predictions.", call. = FALSE)

    # Average across posterior samples to get P(class = 1)
    prob_class_1 <- rowMeans(preds)
    prob_class_0 <- 1 - prob_class_1

    prob_matrix <- cbind(prob_class_0, prob_class_1)
    colnames(prob_matrix) <- object$lvl # Use levels from parsnip object
    return(tibble::as_tibble(prob_matrix))
  }

  stop("`predict_prob_bnns` called for a model that is not a classification model.", call. = FALSE)
}

#' @importFrom tibble tibble
#' @keywords internal
predict_class_bnns <- function(object, new_data) {
  # Leverage the probability wrapper to get class probabilities
  prob_tbl <- predict_prob_bnns(object, new_data)
  
  # Find the index of the maximum probability for each observation
  max_idx <- max.col(prob_tbl, ties.method = "random")
  
  # Extract corresponding factor levels
  pred_classes <- factor(colnames(prob_tbl)[max_idx], levels = object$lvl)
  tibble::tibble(.pred_class = pred_classes)
}

#' @importFrom tibble tibble
#' @keywords internal
predict_numeric_bnns <- function(object, new_data) {
  preds <- predict(object$fit, newdata = new_data)
  
  # Average across the posterior samples to get a single point estimate
  mean_preds <- rowMeans(preds)
  tibble::tibble(.pred = mean_preds)
}

# Internal helpers to allow mocking in testthat
sys_which <- function(x) base::Sys.which(x)
has_namespace <- function(x) base::requireNamespace(x, quietly = TRUE)
ocl_platforms <- function() OpenCL::oclPlatforms()
ocl_devices <- function(p) OpenCL::oclDevices(p)
sys_system2 <- function(...) base::system2(...)

#' OpenCL Diagnostic Information
#'
#' This helper function lists the available OpenCL platforms and devices
#' on your system. It is useful for determining the correct \code{opencl_ids}
#' to pass to \code{bnns()} when using GPU acceleration.
#'
#' @details The function first checks if the \code{clinfo} system command is available.
#' If not, it falls back to looking for the \code{OpenCL} R package to retrieve the
#' platforms and devices.
#'
#' @return Invoked for its side effect of printing OpenCL diagnostic information.
#' @export
opencl_diagnostics <- function() {
  clinfo_path <- sys_which("clinfo")
  if (nchar(clinfo_path) > 0) {
    cat("Found 'clinfo' system command at:", clinfo_path, "\n\n")
    sys_system2(clinfo_path, args = "-l")
  } else if (has_namespace("OpenCL")) {
    cat("Using 'OpenCL' R package for diagnostics:\n\n")
    platforms <- ocl_platforms()
    if (length(platforms) == 0) {
      message("No OpenCL platforms found.")
    } else {
      for (p in seq_along(platforms)) {
        cat(sprintf("Platform #%d:\n", p - 1))
        print(platforms[[p]])
        
        devices <- tryCatch(ocl_devices(platforms[[p]]), error = function(e) list())
        for (d in seq_along(devices)) {
          cat(sprintf("  `-- Device #%d:\n", d - 1))
          print(devices[[d]])
        }
      }
    }
  } else {
    message("Both the 'clinfo' system command and 'OpenCL' R package are missing.\n",
            "To view your OpenCL devices from R, please run:\n\n",
            "  install.packages('OpenCL')\n\n",
            "Or install 'clinfo' on your system.")
  }
  invisible()
}

#' Parameter functions for Bayesian Neural Networks
#'
#' These functions provide \code{dials} parameter objects for tuning
#' the hyperparameters of Bayesian Neural Networks.
#'
#' @param range A two-element vector holding the defaults for the smallest and largest possible values.
#' @param trans A \code{trans} object from the \code{scales} package, could be \code{NULL}.
#' @param values A character vector of possible values.
#' @return A \code{quant_param} or \code{qual_param} object from the \code{dials} package.
#' @name bnns_params
NULL

#' @rdname bnns_params
#' @export
L <- function(range = c(1L, 5L), trans = NULL) {
  if (!requireNamespace("dials", quietly = TRUE)) {
    stop("The 'dials' package is required for this parameter function.", call. = FALSE)
  }
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(L = "Number of Hidden Layers"),
    finalize = NULL
  )
}

#' @rdname bnns_params
#' @export
warmup <- function(range = c(100L, 1000L), trans = NULL) {
  if (!requireNamespace("dials", quietly = TRUE)) {
    stop("The 'dials' package is required for this parameter function.", call. = FALSE)
  }
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(warmup = "Warmup Iterations"),
    finalize = NULL
  )
}

#' @rdname bnns_params
#' @export
chains <- function(range = c(1L, 4L), trans = NULL) {
  if (!requireNamespace("dials", quietly = TRUE)) {
    stop("The 'dials' package is required for this parameter function.", call. = FALSE)
  }
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(chains = "Number of Chains"),
    finalize = NULL
  )
}

#' @rdname bnns_params
#' @export
iter <- function(range = c(500L, 2000L), trans = NULL) {
  if (!requireNamespace("dials", quietly = TRUE)) {
    stop("The 'dials' package is required for this parameter function.", call. = FALSE)
  }
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(iter = "Total Iterations"),
    finalize = NULL
  )
}

#' @rdname bnns_params
#' @export
nodes <- function(range = c(1L, 64L), trans = NULL) {
  if (!requireNamespace("dials", quietly = TRUE)) {
    stop("The 'dials' package is required for this parameter function.", call. = FALSE)
  }
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(nodes = "Nodes per Layer"),
    finalize = NULL
  )
}

#' @rdname bnns_params
#' @export
act_fn <- function(values = c("tanh", "sigmoid", "softplus", "relu", "linear")) {
  if (!requireNamespace("dials", quietly = TRUE)) {
    stop("The 'dials' package is required for this parameter function.", call. = FALSE)
  }
  dials::new_qual_param(
    type = "character",
    values = values,
    label = c(act_fn = "Activation Function")
  )
}
