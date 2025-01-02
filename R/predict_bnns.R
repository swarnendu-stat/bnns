#' Predict Method for \code{"bnns"} Objects
#'
#' Generates predictions from a fitted Bayesian Neural Network (BNN) model.
#'
#' @param object An object of class \code{"bnns"}, typically the result of a call to \code{\link{bnns.default}}.
#' @param newdata A matrix or data frame of new input data for which predictions are required. If \code{NULL}, predictions are made on the training data used to fit the model.
#' @param ... Additional arguments (currently not used).
#'
#' @return A matrix/array of predicted values(regression)/probabilities(classification) where first dimension corresponds to the rows of \code{newdata} or the training data if \code{newdata} is \code{NULL}. Second dimension corresponds to the number of posterior samples. In case of `out_act_fn = 3`, the third dimension corresponds to the class.
#'
#' @details
#' This function uses the posterior distribution from the Stan model in the \code{bnns} object to compute predictions for the provided input data.
#'
#' @examples
#' # Example usage:
#' data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
#' model <- bnns(y ~ -1 + x1 + x2,
#'   data = data, L = 1, nodes = 2, act_fn = 2,
#'   iter = 1e2, warmup = 5e1, chains = 1
#' )
#' new_data <- data.frame(x1 = runif(5), x2 = runif(5))
#' predictions <- predict(model, newdata = new_data)
#' print(predictions)
#'
#' @seealso \code{\link{bnns}}, \code{\link{print.bnns}}
#'
#' @export

predict.bnns <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    test_x <- object$data$X
  } else {
    if (!is.null(object$formula)) {
      test_x <- stats::model.matrix(stats::delete.response(stats::terms.formula(object$formula, data = newdata)), newdata)
    } else {
      test_x <- as.matrix(newdata)
    }
    if(object$normalize){
      test_x <- sweep(test_x, 2, object$x_mean, "-") / object$x_sd
    }
  }

  list_of_draws <- rstan::extract(object$fit)

  if (object$data$out_act_fn == 3) {
    predictions <- array(dim = c(nrow(test_x), dim(list_of_draws$w1)[1], object$data$K))
    for (l in 1:object$data$L) {
      eval(parse(text = paste0("a", l, " <- z", l, " <- array(dim = c(dim(list_of_draws$w1)[1], nrow(test_x), object$data$nodes[", l, "]))")))
    }

    for (i in seq_len(dim(list_of_draws$w1)[1])) {
      z1[i, , ] <- test_x %*% list_of_draws$w1[i, , ] + matrix(rep(list_of_draws$b1[i, ], nrow(test_x)), nrow = nrow(test_x), byrow = TRUE)
      # Activation functions (1 = ta3H, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      if (object$data$act_fn[1] == 1) {
        a1[i, , ] <- tanh(z1[i, , ])
      } else if (object$data$act_fn[1] == 2) {
        a1[i, , ] <- sigmoid(z1[i, , ])
      } else if (object$data$act_fn[1] == 3) {
        a1[i, , ] <- relu(z1[i, , ])
      } else if (object$data$act_fn[1] == 4) {
        a1[i, , ] <- softplus(z1[i, , ])
      }
    }

    if (object$data$L > 1) {
      for (l in 2:object$data$L) {
        for (i in seq_len(dim(list_of_draws$w1)[1])) {
          id3 <- paste0(l, "[", i, ", , ]")
          id3_1 <- paste0(l - 1, "[", i, ", , ]")
          id2 <- paste0(l, "[", i, ", ]")
          eval(parse(text = paste0("z", id3, "<- a", id3_1, "%*% list_of_draws$w", id3, "  + matrix(rep(list_of_draws$b", id2, ", nrow(test_x)), nrow = nrow(test_x), byrow = TRUE)")))
          if (object$data$act_fn[l] == 1) {
            eval(parse(text = paste0("a", id3, " <- tanh(z", id3, ")")))
          } else if (object$data$act_fn[l] == 2) {
            eval(parse(text = paste0("a", id3, " <- sigmoid(z", id3, ")")))
          } else if (object$data$act_fn[l] == 3) {
            eval(parse(text = paste0("a", id3, " <- softplus(z", id3, ")")))
          } else if (object$data$act_fn[l] == 4) {
            eval(parse(text = paste0("a", id3, " <- relu(z", id3, ")")))
          }
        }
      }
    }

    for (i in seq_len(dim(predictions)[2])) {
      id <- paste0(object$data$L, "[", i, ", , ]")
      eval(parse(text = paste0("predictions[,", i, ", ] <- a", id, " %*% list_of_draws$w_out[", i, ", , ] + matrix(rep(list_of_draws$b_out[", i, ", ], nrow(test_x)), nrow = nrow(test_x), byrow = TRUE)")))
    }

    predictions <- softmax_3d(predictions)
  } else {
    predictions <- matrix(nrow = nrow(test_x), ncol = length(list_of_draws$b_out))
    for (l in 1:object$data$L) {
      eval(parse(text = paste0("a", l, " <- z", l, " <- array(dim = c(length(list_of_draws$b_out), nrow(test_x), object$data$nodes[", l, "]))")))
    }

    for (i in seq_len(ncol(predictions))) {
      z1[i, , ] <- test_x %*% list_of_draws$w1[i, , ] + matrix(rep(list_of_draws$b1[i, ], nrow(test_x)), nrow = nrow(test_x), byrow = TRUE)
      # Activation functions (1 = ta3H, 2 = sigmoid, 3 = ReLU, 4 = softplus)
      if (object$data$act_fn[1] == 1) {
        a1[i, , ] <- tanh(z1[i, , ])
      } else if (object$data$act_fn[1] == 2) {
        a1[i, , ] <- sigmoid(z1[i, , ])
      } else if (object$data$act_fn[1] == 3) {
        a1[i, , ] <- relu(z1[i, , ])
      } else if (object$data$act_fn[1] == 4) {
        a1[i, , ] <- softplus(z1[i, , ])
      }
    }

    if (object$data$L > 1) {
      for (l in 2:object$data$L) {
        for (i in seq_len(ncol(predictions))) {
          id3 <- paste0(l, "[", i, ", , ]")
          id3_1 <- paste0(l - 1, "[", i, ", , ]")
          id2 <- paste0(l, "[", i, ", ]")
          eval(parse(text = paste0("z", id3, "<- a", id3_1, "%*% list_of_draws$w", id3, "  + matrix(rep(list_of_draws$b", id2, ", nrow(test_x)), nrow = nrow(test_x), byrow = TRUE)")))
          if (object$data$act_fn[l] == 1) {
            eval(parse(text = paste0("a", id3, " <- tanh(z", id3, ")")))
          } else if (object$data$act_fn[l] == 2) {
            eval(parse(text = paste0("a", id3, " <- sigmoid(z", id3, ")")))
          } else if (object$data$act_fn[l] == 3) {
            eval(parse(text = paste0("a", id3, " <- softplus(z", id3, ")")))
          } else if (object$data$act_fn[l] == 4) {
            eval(parse(text = paste0("a", id3, " <- relu(z", id3, ")")))
          }
        }
      }
    }

    for (i in seq_len(ncol(predictions))) {
      id <- paste0(object$data$L, "[", i, ", , ]")
      eval(parse(text = paste0("predictions[,", i, "] <- a", id, " %*% matrix(list_of_draws$w_out[", i, ", ], ncol = 1) + list_of_draws$b_out[", i, "]")))
    }

    # Activation function for the output layer (1 = linear, 2 = sigmoid, 3 = softmax)
    if (object$data$out_act_fn == 2) {
      predictions <- sigmoid(predictions)
    }
  }
  return(predictions)
}
