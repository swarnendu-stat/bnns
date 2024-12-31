#' Generic Function for Fitting Bayesian Neural Network Models
#'
#' This is a generic function for fitting Bayesian Neural Network (BNN) models. It dispatches to methods based on the class of the input data.
#'
#' @param formula A symbolic description of the model to be fitted. The formula should specify the response variable and predictors (e.g., \code{y ~ x1 + x2}). \code{y} must be continuous for regression (`out_act_fn = 1`), numeric 0/1 for binary classification (`out_act_fn = 2`), and factor with at least 3 levels for multi-classification (`out_act_fn = 3`).
#' @param data A data frame or list containing the variables in the model. Default is an empty list.
#' @param L An integer specifying the number of hidden layers in the neural network. Default is 1.
#' @param nodes An integer or vector specifying the number of nodes in each hidden layer. If a single value is provided, it is applied to all layers. Default is 16.
#' @param act_fn An integer or vector specifying the activation function(s) for the hidden layers. Options are:
#'   \itemize{
#'     \item \code{1} for tanh
#'     \item \code{2} for sigmoid (default)
#'     \item \code{3} for ReLU
#'     \item \code{4} for softplus
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
#' model <- bnns(y ~ -1 + x1 + x2, data = data, L = 1, nodes = 2, act_fn = 1,
#' iter = 1e2, warmup = 5e1, chains = 1)
#'
#' # See the documentation for bnns.default for more details on the default implementation.
#'
#' @seealso \code{\link{bnns.default}}
#'
#' @export

bnns <- function(formula, data = list(), L = 1, nodes = 16,
                 act_fn = 2, out_act_fn = 1, iter = 1e3, warmup = 2e2,
                 thin = 1, chains = 2, cores = 2, seed = 123, ...){
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
#'     \item \code{3} for ReLU
#'     \item \code{4} for softplus
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
#' model <- bnns:::bnns_train(train_x, train_y, L = 1, nodes = 2, act_fn = 2,
#' iter = 1e2, warmup = 5e1, chains = 1)
#'
#' # Access Stan model fit
#' model$fit
#'
#' @seealso \code{\link[rstan]{stan}}
#' @keywords internal

bnns_train <- function(train_x, train_y, L = 1, nodes = 16,
                         act_fn = 2, out_act_fn = 1, iter = 1e3, warmup = 2e2,
                         thin = 1, chains = 2, cores = 2, seed = 123, ...){
  stopifnot("Argument train_x is missing" = !missing(train_x))
  stopifnot("Argument train_y is missing" = !missing(train_y))

  if(out_act_fn == 3){
    stopifnot("train_y must be a factor" = is.factor(train_y))
    stopifnot("train_y must have at least 3 levels" = length(levels(train_y)) >= 3)
    stan_data <- list(
      n = nrow(train_x),                    # Number of observations
      m = ncol(train_x),                    # Number of features
      L = L,
      nodes = nodes,                                # Number of layers
      X = train_x,                          # Input matrix
      y = as.numeric(train_y),                          # Output vector
      K = length(unique(train_y)),
      act_fn = act_fn,         # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn            # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  }else if(out_act_fn == 2){
    stopifnot("train_y must have only 0/1 values" = all(train_y %in% c(0, 1)))
    stan_data <- list(
      n = nrow(train_x),                    # Number of observations
      m = ncol(train_x),                    # Number of features
      L = L,
      nodes = nodes,                                # Number of layers
      X = train_x,                          # Input matrix
      y = train_y,                          # Output vector
      act_fn = act_fn,         # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn            # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  }else{
    stan_data <- list(
      n = nrow(train_x),                    # Number of observations
      m = ncol(train_x),                    # Number of features
      L = L,
      nodes = nodes,                                # Number of layers
      X = train_x,                          # Input matrix
      y = train_y,                          # Output vector
      act_fn = act_fn,         # Activation functions (1 = tanh, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      out_act_fn = out_act_fn            # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    )
  }

  est <- list()
  pars <- c(paste0("w", 1:L), paste0("b", 1:L), "w_out", "b_out")
  if(out_act_fn == 1){ pars <- c(pars, "sigma") }
  est$fit <-  suppressWarnings(rstan::stan(model_code = generate_stan_code(num_layers = L, nodes = nodes, out_act_fn = out_act_fn), data = stan_data, include = TRUE,
                                           pars = pars,
                                           iter = iter, warmup = warmup, thin = thin, chains = chains, cores = cores,
                                           init = 0, seed = seed, verbose = TRUE, refresh = 0))

  est$call <- match.call()
  est$data <- stan_data
  class(est) <- "bnns"
  return(est)
}

#' Bayesian Neural Network Model Using Formula(default) Interface
#'
#' Fits a Bayesian Neural Network (BNN) model using a formula interface. The function parses the formula and data to create the input feature matrix and target vector, then fits the model using \code{\link{bnns.default}}.
#'
#' @param formula A symbolic description of the model to be fitted. The formula should specify the response variable and predictors (e.g., \code{y ~ x1 + x2}).
#' @param data A data frame or list containing the variables in the model. Default is an empty list.
#' @param L An integer specifying the number of hidden layers in the neural network. Default is 1.
#' @param nodes An integer or vector specifying the number of nodes in each hidden layer. If a single value is provided, it is applied to all layers. Default is 16.
#' @param act_fn An integer or vector specifying the activation function(s) for the hidden layers. Options are:
#'   \itemize{
#'     \item \code{1} for tanh
#'     \item \code{2} for sigmoid (default)
#'     \item \code{3} for ReLU
#'     \item \code{4} for softplus
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
#' model <- bnns(y ~ -1 + x1 + x2, data = data, L = 1, nodes = 2, act_fn = 3,
#' iter = 1e2, warmup = 5e1, chains = 1)
#'
#' @export

bnns.default <- function(formula, data=list(), L = 1, nodes = 16,
                         act_fn = 2, out_act_fn = 1, iter = 1e3, warmup = 2e2,
                         thin = 1, chains = 2, cores = 2, seed = 123, ...) {
  if (missing(formula) || missing(data)) {
    stop("Both 'formula' and 'data' must be provided.")
  }

  mf <- stats::model.frame(formula=formula, data=data)
  train_x <- stats::model.matrix(attr(mf, "terms"), data=mf)
  train_y <- stats::model.response(mf)
  est <- bnns_train(train_x = train_x, train_y = train_y, L = L, nodes = nodes,
                      act_fn = act_fn, out_act_fn = out_act_fn, iter = iter,
                      warmup = warmup, thin = thin, chains = chains,
                      cores = cores, seed = seed, ...)
  est$call <- match.call()
  est$formula <- formula
  return(est)
}
