#' Print Method for \code{"bnns"} Objects
#'
#' Displays a summary of a fitted Bayesian Neural Network (BNN) model, including the function call and the Stan fit details.
#'
#' @param x An object of class \code{"bnns"}, typically the result of a call to \code{\link{bnns.default}}.
#' @param ... Additional arguments (currently not used).
#'
#' @return The function is called for its side effects and does not return a value. It prints the following:
#'   \itemize{
#'     \item The function call used to generate the \code{"bnns"} object.
#'     \item A summary of the Stan fit object stored in \code{x$fit}.
#'   }
#'
#' @examples
#' # Example usage:
#' data <- data.frame(x1 = runif(10), x2 = runif(10), y = rnorm(10))
#' model <- bnns(y ~ -1 + x1 + x2,
#'   data = data, L = 1, nodes = 2, act_fn = 2,
#'   iter = 1e2, warmup = 5e1, chains = 1
#' )
#' print(model)
#'
#' @seealso \code{\link{bnns}}, \code{\link{summary.bnns}}
#'
#' @export

print.bnns <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nStan fit:\n")
  print(x$fit)
}
