#' Save a fitted bnns model to disk
#'
#' @param object A fitted \code{bnns} object.
#' @param file A character string specifying the path where the model should be saved (usually ending in .rds).
#' @export
save_bnns <- function(object, file) {
  if (!inherits(object, "bnns")) {
    stop("Object must be of class 'bnns'.", call. = FALSE)
  }
  saveRDS(object, file)
  invisible(file)
}

#' Load a fitted bnns model from disk
#'
#' @param file A character string specifying the path to the saved model.
#' @return A fitted \code{bnns} object.
#' @export
load_bnns <- function(file) {
  object <- readRDS(file)
  if (!inherits(object, "bnns")) {
    stop("The file does not contain a valid 'bnns' object.", call. = FALSE)
  }
  return(object)
}