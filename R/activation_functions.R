#' sigmoid transformation
#'
#' @param x A numeric vector or matrix on which sigmoid transformation is going to be applied.........................
#' @returns A numeric vector or matrix after sigmoid transformation.
#' @examples
#' \dontrun{
#' sigmoid(matrix(1:4), nrow = 2)
#' }
#' @export

sigmoid <- function(x){
  1/(1 + exp(-x))
}

#' softplus transformation
#'
#' @param x A numeric vector or matrix on which softplus transformation is going to be applied.........................
#' @returns A numeric vector or matrix after softplus transformation.
#' @examples
#' \dontrun{
#' softplus(matrix(1:4), nrow = 2)
#' }
#' @export

softplus <- function(x){
  log(1 + exp(x))
}

#' relu transformation
#'
#' @param x A numeric vector or matrix on which relu transformation is going to be applied.........................
#' @returns A numeric vector or matrix after relu transformation.
#' @examples
#' \dontrun{
#' relu(matrix(1:4), nrow = 2)
#' }
#' @export

relu <- function(x){
  if(is.null(dim(x))){
    return(pmax(0, x))
  }else{
    return(matrix(pmax(0, x), nrow = nrow(x), ncol = ncol(x)))
  }
}

#' Apply Softmax Function to a 3D Array
#'
#' This function applies the softmax transformation along the third dimension
#' of a 3D array. The softmax function converts raw scores into probabilities
#' such that they sum to 1 for each slice along the third dimension.
#'
#' @param x A 3D array. The input array on which the softmax function will be applied.
#'
#' @return A 3D array of the same dimensions as `x`, where the values along the
#'   third dimension are transformed using the softmax function.
#'
#' @details
#' The softmax transformation is computed as:
#' \deqn{\text{softmax}(x_{ijk}) = \frac{\exp(x_{ijk})}{\sum_{l} \exp(x_{ijl})}}
#' This is applied for each pair of indices `(i, j)` across the third dimension `(k)`.
#'
#' The function processes the input array slice-by-slice for the first two dimensions
#' `(i, j)`, normalizing the values along the third dimension `(k)` for each slice.
#'
#' @examples
#' \dontrun{
#' # Example: Apply softmax to a 3D array
#' x <- array(runif(24), dim = c(2, 3, 4))  # Random 3D array (2x3x4)
#' softmax_result <- softmax_3d(x)
#'}
#'
#' @export
softmax_3d <- function(x){
  out <- x
  for(i in 1:dim(x)[1]){
    for(j in 1:dim(x)[2]){
      out[i, j, ] <- exp(x[i, j, ])/sum(exp(x[i, j, ]))
    }
  }
  return(out)
}
