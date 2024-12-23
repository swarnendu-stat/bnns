#' tanh transformation
#'
#' @param x A numeric vector or matrix on which tanh transformation is going to be applied.........................
#' @returns A numeric vector or matrix after tanh transformation.
#' @examples
#' tanh(matrix(1:4), nrow = 2)

tanh <- function(x){
  (exp(x) - exp(-x))/(exp(x) + exp(-x))
}

#' sigmoid transformation
#'
#' @param x A numeric vector or matrix on which sigmoid transformation is going to be applied.........................
#' @returns A numeric vector or matrix after sigmoid transformation.
#' @examples
#' sigmoid(matrix(1:4), nrow = 2)

sigmoid <- function(x){
  1/(1 + exp(-x))
}

#' softplus transformation
#'
#' @param x A numeric vector or matrix on which softplus transformation is going to be applied.........................
#' @returns A numeric vector or matrix after softplus transformation.
#' @examples
#' softplus(matrix(1:4), nrow = 2)

softplus <- function(x){
  log(1 + exp(x))
}

#' relu transformation
#'
#' @param x A numeric vector or matrix on which relu transformation is going to be applied.........................
#' @returns A numeric vector or matrix after relu transformation.
#' @examples
#' relu(matrix(1:4), nrow = 2)

relu <- function(x){
  if(is.null(dim(x))){
    return(pmax(0, x))
  }else{
    return(matrix(pmax(0, x), nrow = nrow(x), ncol = ncol(x)))
  }
}
