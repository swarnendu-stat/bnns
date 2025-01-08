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
#' @param out_act_fn An integer specifying the activation function for the output layer. Options are:
#'   \itemize{
#'     \item \code{1} for linear (default)
#'     \item \code{2} for sigmoid
#'     \item \code{3} for softmax
#'   }
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
#' # Example usage with formula interface:
#' data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
#' model <- bnns(y ~ -1 + x1 + x2,
#'   data = data, L = 1, nodes = 2, act_fn = 1,
#'   iter = 1e2, warmup = 5e1, chains = 1
#' )
#'
#' # See the documentation for bnns.default for more details on the default implementation.
#'
#' @seealso \code{\link{bnns.default}}
#'
#' @export

bnns <- function(formula, data, L = 1, nodes = rep(2, L),
                 act_fn = rep(2, L), out_act_fn = 1, iter = 1e3, warmup = 2e2,
                 thin = 1, chains = 2, cores = 2, seed = 123, prior_weights = NULL,
                 prior_bias = NULL, prior_sigma = NULL, verbose = FALSE,
                 refresh = max(iter / 10, 1), normalize = TRUE, ...) {
  UseMethod("bnns")
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
#' @param out_act_fn An integer specifying the activation function for the output layer. Options are:
#'   \itemize{
#'     \item \code{1} for linear (default)
#'     \item \code{2} for sigmoid
#'     \item \code{3} for softmax
#'   }
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
#' # Example usage:
#' train_x <- matrix(runif(20), nrow = 10, ncol = 2)
#' train_y <- rnorm(10)
#' model <- bnns:::bnns_train(train_x, train_y,
#'   L = 1, nodes = 2, act_fn = 2,
#'   iter = 1e2, warmup = 5e1, chains = 1
#' )
#'
#' # Access Stan model fit
#' model$fit
#'
#' @seealso \code{\link[rstan]{stan}}
#'
#' @export
#' @keywords internal

bnns_train <- function(train_x, train_y, L = 1, nodes = rep(2, L),
                       act_fn = rep(2, L), out_act_fn = 1, iter = 1e3, warmup = 2e2,
                       thin = 1, chains = 2, cores = 2, seed = 123, prior_weights = NULL,
                       prior_bias = NULL, prior_sigma = NULL, verbose = FALSE,
                       refresh = max(iter / 10, 1), normalize = TRUE, ...) {
  stopifnot("Argument train_x is missing" = !missing(train_x))
  stopifnot("Argument train_y is missing" = !missing(train_y))
  stopifnot("L must be a positive integer" = ((L %% 1 == 0) & (sign(L) == 1)))
  stopifnot("nodes must be of length L" = length(nodes) == L)
  stopifnot("nodes must be positive integer(s)" = (all(nodes %% 1 == 0) & all(sign(nodes) == 1)))
  stopifnot("act_fn must be of length L" = length(act_fn) == L)
  stopifnot("act_fn must be a sequence of 1/2/3/4/5" = all(act_fn %in% 1:5))

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

  supported_distributions <- c("normal", "uniform", "cauchy")
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

  # Replace PRIOR_SPECIFICATION with the appropriate Stan syntax
  prior_specification_weights <- switch(prior_weights$dist,
    normal = sprintf("normal(%f, %f)", prior_weights$params$mean, prior_weights$params$sd),
    uniform = sprintf("uniform(%f, %f)", prior_weights$params$alpha, prior_weights$params$beta),
    cauchy = sprintf("cauchy(%f, %f)", prior_weights$params$mu, prior_weights$params$sigma)
  )

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
      nodes = nodes, # Number of layers
      X = train_x, # Input matrix
      y = as.numeric(train_y), # Output vector
      K = length(unique(train_y)),
      act_fn = act_fn, # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  } else if (out_act_fn == 2) {
    stopifnot("train_y must have only 0/1 values" = all(train_y %in% c(0, 1)))
    stan_data <- list(
      n = nrow(train_x), # Number of observations
      m = ncol(train_x), # Number of features
      L = L,
      nodes = nodes, # Number of layers
      X = train_x, # Input matrix
      y = train_y, # Output vector
      act_fn = act_fn, # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  } else {
    stan_data <- list(
      n = nrow(train_x), # Number of observations
      m = ncol(train_x), # Number of features
      L = L,
      nodes = nodes, # Number of layers
      X = train_x, # Input matrix
      y = train_y, # Output vector
      act_fn = act_fn, # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  }

  est <- list()
  pars <- c(paste0("w", 1:L), paste0("b", 1:L), "w_out", "b_out")
  if (out_act_fn == 1) {
    pars <- c(pars, "sigma")
  }

  stan_model <- gsub("PRIOR_WEIGHT", prior_specification_weights, generate_stan_code(num_layers = L, nodes = nodes, out_act_fn = out_act_fn)) |>
    gsub(x = _, pattern = "PRIOR_BIAS", replacement = prior_specification_bias)

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

  est$fit <- suppressWarnings(rstan::stan(
    model_code = stan_model, data = stan_data, include = TRUE,
    pars = pars,
    iter = iter, warmup = warmup, thin = thin, chains = chains, cores = cores,
    init = 0, seed = seed, verbose = verbose, refresh = refresh
  ))

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
#' @param out_act_fn An integer specifying the activation function for the output layer. Options are:
#'   \itemize{
#'     \item \code{1} for linear (default)
#'     \item \code{2} for sigmoid
#'     \item \code{3} for softmax
#'   }
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
#' # Example usage with formula interface:
#' data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
#' model <- bnns(y ~ -1 + x1 + x2,
#'   data = data, L = 1, nodes = 2, act_fn = 3,
#'   iter = 1e2, warmup = 5e1, chains = 1
#' )
#'
#' @export

bnns.default <- function(formula, data, L = 1, nodes = rep(2, L),
                         act_fn = rep(2, L), out_act_fn = 1, iter = 1e3, warmup = 2e2,
                         thin = 1, chains = 2, cores = 2, seed = 123, prior_weights = NULL,
                         prior_bias = NULL, prior_sigma = NULL, verbose = FALSE,
                         refresh = max(iter / 10, 1), normalize = TRUE, ...) {
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
  est <- bnns_train(
    train_x = train_x, train_y = train_y, L = L, nodes = nodes,
    act_fn = act_fn, out_act_fn = out_act_fn, iter = iter,
    warmup = warmup, thin = thin, chains = chains,
    cores = cores, seed = seed, prior_weights = prior_weights,
    prior_bias = prior_bias, prior_sigma = prior_sigma,
    verbose = verbose, refresh = refresh, normalize = normalize, ...
  )
  est$call <- match.call()
  est$formula <- formula
  return(est)
}
