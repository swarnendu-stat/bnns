#' Predictions from a fitted Bayesian Neural Network
#'
#' @param object A fitted \code{bnns} model object.
#' @param newdata A data frame containing new data for prediction. If not provided,
#'   the predictions will be generated using the training data.
#' @param type Character string indicating the type of prediction. 
#'   Options are \code{"samples"} (default), \code{"mean"}, \code{"median"}, \code{"quantile"}, 
#'   \code{"prob"} (class probabilities), or \code{"class"} (predicted class labels).
#' @param quantiles Numeric vector of probabilities used when \code{type = "quantile"}.
#'   Default is \code{c(0.025, 0.975)}.
#' @param ... Additional arguments passed to internal prediction methods.
#'
#' @return 
#' \itemize{
#'   \item For \code{type = "samples"}: A matrix (regression/binary) or 3D array (multiclass) of posterior predictions.
#'   \item For \code{type = "mean"} or \code{"median"}: A vector or matrix of aggregated predictions. For classification tasks, \code{type = "mean"} returns the posterior mean class probabilities.
#'   \item For \code{type = "quantile"}: A matrix or array of quantiles.
#'   \item For \code{type = "prob"}: A matrix of class probabilities (for classification models).
#'   \item For \code{type = "class"}: A vector of predicted class labels (for classification models).
#' }
#' @export
predict.bnns <- function(object, newdata = NULL, 
                         type = c("samples", "mean", "median", "quantile", "prob", "class"),
                         quantiles = c(0.025, 0.975), ...) {
  
  # 1. Validate the 'type' argument
  type <- match.arg(type)
  
  if (type %in% c("prob", "class") && object$data$out_act_fn == 1) {
    stop("type = '", type, "' is only applicable for classification models.", call. = FALSE)
  }
  
  # 2. Generate raw posterior samples 
  raw_samples <- .predict_raw(object, newdata, ...)
  
  # Return the full posterior distributions immediately if requested
  if (type == "samples") {
    return(raw_samples)
  }
  
  # 3. Check dimensions 
  # - Regression/Binary: typically [n_obs, n_samples]
  # - Multiclass: typically [n_obs, n_samples, n_classes]
  is_3d <- length(dim(raw_samples)) == 3
  
  # 4. Aggregate based on the requested type
  if (is_3d) {
    # Multiclass aggregation
    res <- switch(type,
      mean     = apply(raw_samples, c(1, 3), mean),
      median   = apply(raw_samples, c(1, 3), median),
      quantile = apply(raw_samples, c(1, 3), stats::quantile, probs = quantiles),
      prob     = apply(raw_samples, c(1, 3), mean),
      class    = {
        p <- apply(raw_samples, c(1, 3), mean)
        idx <- max.col(p, ties.method = "random")
        if (!is.null(object$levels)) object$levels[idx] else idx
      }
    )
    
    if (type %in% c("prob", "mean", "median") && !is.null(object$levels)) {
      colnames(res) <- object$levels
    }
  } else {
    # Regression / Binary aggregation
    res <- switch(type,
      mean     = rowMeans(raw_samples),
      median   = apply(raw_samples, 1, median),
      quantile = apply(raw_samples, 1, stats::quantile, probs = quantiles),
      prob     = {
        p <- rowMeans(raw_samples)
        out <- cbind("0" = 1 - p, "1" = p)
        if (!is.null(object$levels) && length(object$levels) == 2) colnames(out) <- object$levels
        out
      },
      class    = {
        p <- rowMeans(raw_samples)
        idx <- ifelse(p > 0.5, 2, 1)
        if (!is.null(object$levels) && length(object$levels) == 2) object$levels[idx] else idx - 1
      }
    )
    
    # Format quantiles into a cleaner matrix layout (rows = observations, cols = quantiles)
    if (type == "quantile") {
      res <- t(res)
      colnames(res) <- paste0("Q_", quantiles * 100)
    }
  }
  
  return(res)
}

#' Internal function to perform forward passes using posterior samples
#'
#' @param object A fitted `bnns` model
#' @param newdata A data frame containing new features
#' @param ... Additional arguments (unused)
#' @return A matrix of predictions [obs, samples] or 3D array [obs, samples, classes]
#' @keywords internal
#' @noRd
.predict_raw <- function(object, newdata = NULL, ...) {
  # 1. Prepare input data
  if (is.null(newdata)) {
    X <- object$data$X
  } else {
    if (!is.null(object$terms)) {
      tt <- stats::delete.response(object$terms)
    } else {
      tt <- stats::delete.response(stats::terms(object$formula, data = newdata))
    }
    mf <- stats::model.frame(tt, data = newdata)
    X <- stats::model.matrix(tt, data = mf)
    
    # Apply the exact same normalization used during training
    if (isTRUE(object$normalize)) {
      X <- sweep(X, 2, object$x_mean, "-")
      X <- sweep(X, 2, object$x_sd, "/")
    }
  }

  # 2. Extract configuration and posterior samples
  stan_data <- object$data
  samples <- rstan::extract(object$fit)
  
  # Robustly determine the number of posterior samples across any configuration
  S <- if (!is.null(dim(samples[[1]]))) dim(samples[[1]])[1] else length(samples[[1]])
  N <- nrow(X)             # Number of observations
  L <- stan_data$L         # Number of hidden layers
  
  act_fn <- rep_len(stan_data$act_fn, L)
  out_act_fn <- stan_data$out_act_fn
  
  # Helper functions for activation
  apply_act <- function(z, type) {
    switch(as.character(type),
      "1" = tanh(z),
      "2" = 1 / (1 + exp(-z)),       # sigmoid
      "3" = log1p(exp(z)),           # softplus
      "4" = { z[z < 0] <- 0; z },    # ReLU (modifies in place to preserve matrix dims)
      "5" = z                        # linear
    )
  }
  
  apply_out_act <- function(z, type) {
    switch(as.character(type),
      "1" = z,                       # linear (regression)
      "2" = 1 / (1 + exp(-z)),       # sigmoid (binary classification)
      "3" = {                        # softmax (multiclass)
        # Shift z for numerical stability to prevent exp() overflow
        z_shift <- sweep(z, 1, apply(z, 1, max), "-")
        ez <- exp(z_shift)
        sweep(ez, 1, rowSums(ez), "/")
      }
    )
  }
  
  # 3. Initialize Output Structure
  if (out_act_fn == 3) {
    K <- stan_data$K
    out <- array(NA, dim = c(N, S, K))
  } else {
    out <- matrix(NA, nrow = N, ncol = S)
  }
  
  # 4. Perform the Forward Pass
  for (s in seq_len(S)) {
    A <- X
    
    # Process each hidden layer
    for (l in seq_len(L)) {
      w_l <- samples[[paste0("w", l)]]
      if (length(dim(w_l)) == 3) {
        W_s <- matrix(w_l[s, , ], nrow = ncol(A))
      } else if (length(dim(w_l)) == 2) {
        W_s <- matrix(w_l[s, ], nrow = ncol(A))
      } else {
        W_s <- matrix(w_l[s], nrow = ncol(A))
      }
      
      b_l <- samples[[paste0("b", l)]]
      b_s <- if (length(dim(b_l)) == 2) b_l[s, ] else b_l[s]
      
      Z <- A %*% W_s
      Z <- sweep(Z, 2, b_s, "+")
      
      A <- apply_act(Z, act_fn[l])
    }
    
    # Process output layer
    w_out <- samples[["w_out"]]
    if (length(dim(w_out)) == 3) {
      W_out_s <- matrix(w_out[s, , ], nrow = ncol(A))
    } else if (length(dim(w_out)) == 2) {
      W_out_s <- matrix(w_out[s, ], nrow = ncol(A))
    } else {
      W_out_s <- matrix(w_out[s], nrow = ncol(A))
    }
    
    b_out <- samples[["b_out"]]
    b_out_s <- if (length(dim(b_out)) == 2) b_out[s, ] else b_out[s]
    
    Z_out <- A %*% W_out_s
    Z_out <- sweep(Z_out, 2, b_out_s, "+")
    
    A_out <- apply_out_act(Z_out, out_act_fn)
    
    if (out_act_fn == 3) {
      out[, s, ] <- A_out
    } else {
      out[, s] <- as.vector(A_out)
    }
  }
  
  return(out)
}