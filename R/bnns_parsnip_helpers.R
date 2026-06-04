#' Translate parsnip activation name to bnns code
#' @keywords internal
translate_activation <- function(activation) {
  mapping <- c(
    "tanh"     = 1L,
    "sigmoid"  = 2L,
    "softplus" = 3L,
    "relu"     = 4L,
    "linear"   = 5L
  )
  if (is.character(activation)) {
    activation <- tolower(activation)
    if (any(!activation %in% names(mapping))) {
      rlang::abort(paste0(
        "Unknown activation. ",
        "bnns supports: ", paste(names(mapping), collapse = ", ")
      ))
    }
    return(unname(mapping[activation]))
  }
  as.integer(activation)
}

#' Auto-detect out_act_fn from response variable
#' @keywords internal
detect_output_activation <- function(y) {
  if (is.numeric(y) && !is.factor(y)) {
    return(1L)  # regression
  }
  lvls <- if (is.factor(y)) levels(y) else unique(y)
  if (length(lvls) == 2) {
    return(2L)  # binary
  }
  return(3L)    # multiclass
}