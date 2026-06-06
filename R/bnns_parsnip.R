#' @keywords internal
#' @noRd
post_pred_numeric <- function(results, object) {
  tibble::tibble(.pred = rowMeans(results))
}

#' @keywords internal
#' @noRd
post_pred_class <- function(results, object) {
  pred_probs <- rowMeans(results)
  lvls <- object$fit$levels
  pred_class <- ifelse(pred_probs > 0.5, lvls[2], lvls[1])
  tibble::tibble(.pred_class = factor(pred_class, levels = lvls))
}

#' @keywords internal
#' @noRd
post_pred_prob <- function(results, object) {
  lvls <- object$fit$levels
  if (length(dim(results)) == 2) {
    prob_pos <- rowMeans(results)
    out <- tibble::tibble(1 - prob_pos, prob_pos)
    names(out) <- paste0(".pred_", lvls)
  } else {
    out <- as.data.frame(apply(results, c(1, 3), mean))
    names(out) <- paste0(".pred_", lvls)
  }
  tibble::as_tibble(out)
}

if (getRversion() >= "2.15.1") {
  utils::globalVariables(c("object", "new_data"))
}

#' Register bnns parsnip engine
#'
#' @param force Logical. Force registration even if already registered.
#' @param model Character. The parsnip model name to register. Default is `"mlp"`.
#' @keywords internal
#' @noRd
register_bnns_parsnip <- function(force = FALSE, model = "mlp") {
  
  if (!requireNamespace("parsnip", quietly = TRUE)) return(invisible(NULL))
  
  if (!force) {
    # Check if already registered
    engines <- try(parsnip::show_engines(model), silent = TRUE)
    if (!inherits(engines, "try-error") && "bnns" %in% engines$engine) {
      return(invisible(NULL))
    }
  }
  
  # ===== REGRESSION =====
  
  parsnip::set_model_engine(model, mode = "regression", eng = "bnns")
  parsnip::set_dependency(model, eng = "bnns", pkg = "bnns")
  
  # Map parsnip args -> bnns args
  parsnip::set_model_arg(
    model = model, eng = "bnns",
    parsnip = "hidden_units",
    original = "nodes",
    func = list(pkg = "dials", fun = "hidden_units"),
    has_submodel = FALSE
  )
  
  parsnip::set_model_arg(
    model = model, eng = "bnns",
    parsnip = "epochs",
    original = "iter",
    func = list(pkg = "dials", fun = "epochs"),
    has_submodel = FALSE
  )
  
  parsnip::set_model_arg(
    model = model, eng = "bnns",
    parsnip = "activation",
    original = "act_fn",
    func = list(pkg = "dials", fun = "activation"),
    has_submodel = FALSE
  )
  
  # Encoding: how parsnip should prepare data
  parsnip::set_encoding(
    model = model, eng = "bnns", mode = "regression",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept = FALSE,
      remove_intercept = TRUE,
      allow_sparse_x = FALSE
    )
  )
  
  # Fit specification
  parsnip::set_fit(
    model = model, mode = "regression", eng = "bnns",
    value = list(
      interface = "formula",
      protect = c("formula", "data"),
      func = c(pkg = "bnns", fun = "bnns"),
      defaults = list(
        out_act_fn = 1L,
        L = 1L
      )
    )
  )
  
  # Predict: numeric
  parsnip::set_pred(
    model = model, mode = "regression", eng = "bnns",
    type = "numeric",
    value = list(
      pre = NULL,
      post = post_pred_numeric,
      func = c(pkg = "stats", fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )
  
  # Predict: raw (full posterior samples)
  parsnip::set_pred(
    model = model, mode = "regression", eng = "bnns",
    type = "raw",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(pkg = "stats", fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )
  
  # ===== CLASSIFICATION =====
  
  parsnip::set_model_engine(model, mode = "classification", eng = "bnns")
  parsnip::set_dependency(model, eng = "bnns", pkg = "bnns", mode = "classification")
  
  # Same arg mappings for classification
  parsnip::set_model_arg(
    model = model, eng = "bnns",
    parsnip = "hidden_units",
    original = "nodes",
    func = list(pkg = "dials", fun = "hidden_units"),
    has_submodel = FALSE
  )
  
  parsnip::set_model_arg(
    model = model, eng = "bnns",
    parsnip = "epochs",
    original = "iter",
    func = list(pkg = "dials", fun = "epochs"),
    has_submodel = FALSE
  )
  
  parsnip::set_model_arg(
    model = model, eng = "bnns",
    parsnip = "activation",
    original = "act_fn",
    func = list(pkg = "dials", fun = "activation"),
    has_submodel = FALSE
  )
  
  parsnip::set_encoding(
    model = model, eng = "bnns", mode = "classification",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept = FALSE,
      remove_intercept = TRUE,
      allow_sparse_x = FALSE
    )
  )
  
  parsnip::set_fit(
    model = model, mode = "classification", eng = "bnns",
    value = list(
      interface = "formula",
      protect = c("formula", "data"),
      func = c(pkg = "bnns", fun = "bnns"),
      defaults = list(
        L = 1L
      )
    )
  )
  
  # Predict: class labels
  parsnip::set_pred(
    model = model, mode = "classification", eng = "bnns",
    type = "class",
    value = list(
      pre = NULL,
      post = post_pred_class,
      func = c(pkg = "stats", fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )
  
  # Predict: class probabilities
  parsnip::set_pred(
    model = model, mode = "classification", eng = "bnns",
    type = "prob",
    value = list(
      pre = NULL,
      post = post_pred_prob,
      func = c(pkg = "stats", fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )
}