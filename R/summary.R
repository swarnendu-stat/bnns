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
#' @return None. The function prints the summary to the console.
#'
#' @examples
#' \dontrun{
#' # Fit a Bayesian Neural Network
#' df <- data.frame(x1 = runif(100), x2 = runif(100), y = rnorm(100))
#' model <- bnns(y ~ -1 + x1 + x2, data = df, L = 2, nodes = c(16, 8), act_fn = c(2, 3))
#'
#' # Get a summary of the model
#' summary(model)
#' }
#'
#' @seealso \code{\link{bnns}}, \code{\link{print.bnns}}
#'
#' @export

summary.bnns <- function(object, ...){
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
  if(object$data$out_act_fn == 1){ pars <- c(pars, "sigma") }
  cat("\nPosterior Summary (Key Parameters):\n")
  stan_sum <- rstan::summary(object$fit, pars = pars)$summary
  print(stan_sum)

  cat("\nModel Fit Information:\n")
  cat("Iterations:", object$fit@sim$iter, "\n")
  cat("Warmup:", object$fit@sim$warmup, "\n")
  cat("Thinning:", object$fit@sim$thin, "\n")
  cat("Chains:", object$fit@sim$chains, "\n")

  cat("\nPredictive Performance:\n")
  if(object$data$out_act_fn == 1){
    measure <- measure_cont(obs = object$data$y, pred = predict.bnns(object))
    cat("RMSE (training):", measure$rmse, "\n")
    cat("MAE (training):", measure$mae, "\n")
  }else if(object$data$out_act_fn == 2){
    measure <- measure_bin(obs = object$data$y, pred = predict.bnns(object))
    cat("Confusion matrix (training with 0.5 cutoff):", measure$conf_mat, "\n")
    cat("Accuracy (training with 0.5 cutoff):", measure$accuracy, "\n")
    cat(measure$AUC, "\n")
  }else if(object$data$out_act_fn == 3){
    measure <- measure_cat(obs = factor(object$data$y), pred = predict.bnns(object))
    cat("Log-loss (training):", measure$log_loss, "\n")
    cat("AUC (training):", measure$AUC, "\n")
  }

  cat("\nNotes:\n")
  cat("Check convergence diagnostics for parameters with high R-hat values.\n")
  return(invisible(list("Number of observations" = object$data$n,
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
              "Performance" = measure)))
}
