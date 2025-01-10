#' Summary of a Bayesian Neural Network (BNN) Model
#'
#' Provides a comprehensive summary of a fitted Bayesian Neural Network (BNN) model, including details about the model call, data, network architecture, posterior distributions, and model fitting information.
#'
#' @param object An object of class \code{bnns}, representing a fitted Bayesian Neural Network model.
#' @param ... Additional arguments (currently unused).
#'
#' @details The function prints the following information:
#' \itemize{
#'   \item \strong{Call:} The original function call used to fit the model.
#'   \item \strong{Data Summary:} Number of observations and features in the training data.
#'   \item \strong{Network Architecture:} Structure of the BNN including the number of hidden layers, nodes per layer, and activation functions.
#'   \item \strong{Posterior Summary:} Summarized posterior distributions of key parameters (e.g., weights, biases, and noise parameter).
#'   \item \strong{Model Fit Information:} Bayesian sampling details, including the number of iterations, warmup period, thinning, and chains.
#'   \item \strong{Notes:} Remarks and warnings, such as checks for convergence diagnostics.
#' }
#'
#' @return A list (returned invisibly) containing the following elements:
#' \itemize{
#'   \item \code{"Number of observations"}: The number of observations in the training data.
#'   \item \code{"Number of features"}: The number of features in the training data.
#'   \item \code{"Number of hidden layers"}: The number of hidden layers in the neural network.
#'   \item \code{"Nodes per layer"}: A comma-separated string representing the number of nodes in each hidden layer.
#'   \item \code{"Activation functions"}: A comma-separated string representing the activation functions used in each hidden layer.
#'   \item \code{"Output activation function"}: The activation function used in the output layer.
#'   \item \code{"Stanfit Summary"}: A summary of the Stan model, including key parameter posterior distributions.
#'   \item \code{"Iterations"}: The total number of iterations used for sampling in the Bayesian model.
#'   \item \code{"Warmup"}: The number of iterations used as warmup in the Bayesian model.
#'   \item \code{"Thinning"}: The thinning interval used in the Bayesian model.
#'   \item \code{"Chains"}: The number of Markov chains used in the Bayesian model.
#'   \item \code{"Performance"}: Predictive performance metrics, which vary based on the output activation function.
#' }
#' The function also prints the summary to the console.
#'
#' @examples
#' \donttest{
#' # Fit a Bayesian Neural Network
#' data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
#' model <- bnns(y ~ -1 + x1 + x2,
#'   data = data, L = 1, nodes = 2, act_fn = 2,
#'   iter = 1e1, warmup = 5, chains = 1
#' )
#'
#' # Get a summary of the model
#' summary(model)
#' }
#' @seealso \code{\link{bnns}}, \code{\link{print.bnns}}
#'
#' @export

summary.bnns <- function(object, ...) {
  cat("Call:\n")
  print(object$call)

  cat("\nData Summary:\n")
  cat("Number of observations:", object$data$n, "\n")
  cat("Number of features:", object$data$m, "\n")

  cat("\nNetwork Architecture:\n")
  cat("Number of hidden layers:", object$data$L, "\n")
  cat("Nodes per layer:", paste(object$data$nodes, collapse = ", "), "\n")
  cat("Activation functions:", paste(object$data$act_fn, collapse = ", "), "\n")
  cat("Output activation function:", object$data$out_act_fn, "\n")

  pars <- c("w_out", "b_out")
  if (object$data$out_act_fn == 1) {
    pars <- c(pars, "sigma")
  }
  cat("\nPosterior Summary (Key Parameters):\n")
  stan_sum <- rstan::summary(object$fit, pars = pars)$summary
  print(stan_sum)

  cat("\nModel Fit Information:\n")
  cat("Iterations:", object$fit@sim$iter, "\n")
  cat("Warmup:", object$fit@sim$warmup, "\n")
  cat("Thinning:", object$fit@sim$thin, "\n")
  cat("Chains:", object$fit@sim$chains, "\n")

  cat("\nPredictive Performance:\n")
  if (object$data$out_act_fn == 1) {
    measure <- measure_cont(obs = object$data$y, pred = predict.bnns(object))
    cat("RMSE (training):", measure$rmse, "\n")
    cat("MAE (training):", measure$mae, "\n")
  } else if (object$data$out_act_fn == 2) {
    measure <- measure_bin(obs = object$data$y, pred = predict.bnns(object))
    cat("Confusion matrix (training with 0.5 cutoff):", measure$conf_mat, "\n")
    cat("Accuracy (training with 0.5 cutoff):", measure$accuracy, "\n")
    cat("AUC (training):", measure$AUC, "\n")
  } else if (object$data$out_act_fn == 3) {
    measure <- measure_cat(obs = factor(object$data$y), pred = predict.bnns(object))
    cat("Log-loss (training):", measure$log_loss, "\n")
    cat("AUC (training):", measure$AUC, "\n")
  }

  cat("\nNotes:\n")
  cat("Check convergence diagnostics for parameters with high R-hat values.\n")
  return(invisible(list(
    "Number of observations" = object$data$n,
    "Number of features" = object$data$m,
    "Number of hidden layers" = object$data$L,
    "Nodes per layer" = paste(object$data$nodes, collapse = " , "),
    "Activation functions" = paste(object$data$act_fn, collapse = " , "),
    "Output activation function" = object$data$out_act_fn,
    "Stanfit Summary" = stan_sum,
    "Iterations" = object$fit@sim$iter,
    "Warmup" = object$fit@sim$warmup,
    "Thinning" = object$fit@sim$thin,
    "Chains" = object$fit@sim$chains,
    "Performance" = measure
  )))
}
