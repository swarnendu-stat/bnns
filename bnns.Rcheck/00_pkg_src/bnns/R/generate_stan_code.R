#' Generate Stan code for a Bayesian Neural Network
#'
#' This function dynamically generates the Stan code for a BNN based on the
#' specified architecture and priors.
#'
#' @param num_layers Number of hidden layers.
#' @param nodes Vector of nodes for each hidden layer.
#' @param out_act_fn Integer code for the output activation function.
#' @param prior_weights_dist Character string for the weight prior distribution.
#' @return A character string containing the full Stan model code.
#' @keywords internal
#' @noRd
generate_stan_code <- function(num_layers, nodes, out_act_fn, prior_weights_dist = "normal") {
  # --- Data Block ---
  data_block <- "
data {
  int<lower=1> n; // Number of observations
  int<lower=1> m; // Number of features
  int<lower=1> L; // Number of hidden layers
  array[L] int nodes; // Nodes per layer
  matrix[n, m] X; // Input matrix
  "
  if (out_act_fn == 3) {
    data_block <- paste0(data_block, "  array[n] int y; // Output vector for multiclass\n")
    data_block <- paste0(data_block, "  int<lower=1> K; // Number of classes\n")
  } else if (out_act_fn == 2) {
    data_block <- paste0(data_block, "  array[n] int y; // Output vector for binary\n")
  } else {
    data_block <- paste0(data_block, "  vector[n] y; // Output vector for regression\n")
  }
  data_block <- paste0(data_block, "  array[L] int act_fn; // Activation functions\n")
  data_block <- paste0(data_block, "  int out_act_fn; // Output activation function\n")
  data_block <- paste0(data_block, "}\n")

  # --- Parameters Block ---
  params_block <- "parameters {\n"
  # Hidden layer parameters
  for (l in 1:num_layers) {
    rows <- ifelse(l == 1, "m", paste0("nodes[", l - 1, "]"))
    cols <- paste0("nodes[", l, "]")
    if (prior_weights_dist == "horseshoe") {
      params_block <- paste0(params_block, sprintf("  vector[%s * %s] w%d_raw;\n", rows, cols, l))
      params_block <- paste0(params_block, sprintf("  vector<%s=0>[%s * %s] lambda_w%d;\n", "lower", rows, cols, l))
      params_block <- paste0(params_block, sprintf("  real<%s=0> tau_w%d;\n", "lower", l))
    } else {
      params_block <- paste0(params_block, sprintf("  matrix[%s, %s] w%d;\n", rows, cols, l))
    }
    params_block <- paste0(params_block, sprintf("  vector[%s] b%d;\n", cols, l))
  }
  # Output layer parameters
  rows_out <- paste0("nodes[", num_layers, "]")
  cols_out <- ifelse(out_act_fn == 3, "K", "1")
  if (prior_weights_dist == "horseshoe") {
    params_block <- paste0(params_block, sprintf("  vector[%s * %s] w_out_raw;\n", rows_out, cols_out))
    params_block <- paste0(params_block, sprintf("  vector<%s=0>[%s * %s] lambda_w_out;\n", "lower", rows_out, cols_out))
    params_block <- paste0(params_block, "  real<lower=0> tau_w_out;\n")
  } else {
    if (out_act_fn == 3) {
      params_block <- paste0(params_block, sprintf("  matrix[%s, %s] w_out;\n", rows_out, cols_out))
    } else {
      params_block <- paste0(params_block, sprintf("  vector[%s] w_out;\n", rows_out))
    }
  }
  params_block <- paste0(params_block, sprintf("  vector[%s] b_out;\n", cols_out))
  # Sigma for regression
  if (out_act_fn == 1) {
    params_block <- paste0(params_block, "  real<lower=0> sigma;\n")
  }
  params_block <- paste0(params_block, "}\n")

  # --- Transformed Parameters Block ---
  tparams_block <- ""
  if (prior_weights_dist == "horseshoe") {
    tparams_block <- "transformed parameters {\n"
    # Reconstruct hidden layer weights
    for (l in 1:num_layers) {
      rows <- ifelse(l == 1, "m", paste0("nodes[", l - 1, "]"))
      cols <- paste0("nodes[", l, "]")
      tparams_block <- paste0(tparams_block, sprintf("  matrix[%s, %s] w%d = to_matrix(w%d_raw .* lambda_w%d * tau_w%d, %s, %s);\n", rows, cols, l, l, l, l, rows, cols))
    }
    # Reconstruct output layer weights
    rows_out <- paste0("nodes[", num_layers, "]")
    cols_out <- ifelse(out_act_fn == 3, "K", "1")
    if (out_act_fn == 3) {
      tparams_block <- paste0(tparams_block, sprintf("  matrix[%s, %s] w_out = to_matrix(w_out_raw .* lambda_w_out * tau_w_out, %s, %s);\n", rows_out, cols_out, rows_out, cols_out))
    } else {
      tparams_block <- paste0(tparams_block, sprintf("  vector[%s] w_out = w_out_raw .* lambda_w_out * tau_w_out;\n", rows_out))
    }
    tparams_block <- paste0(tparams_block, "}\n")
  }

  # --- Model Block ---
  model_block <- "model {\n"
  # Priors
  for (l in 1:num_layers) {
    if (prior_weights_dist == "horseshoe") {
      model_block <- paste0(model_block, sprintf("  w%d_raw ~ normal(0, 1);\n", l))
      model_block <- paste0(model_block, sprintf("  lambda_w%d ~ cauchy(0, 1);\n", l))
      model_block <- paste0(model_block, sprintf("  tau_w%d ~ cauchy(0, 1);\n", l))
    } else {
      model_block <- paste0(model_block, sprintf("  to_vector(w%d) ~ PRIOR_WEIGHT;\n", l))
    }
    model_block <- paste0(model_block, sprintf("  b%d ~ PRIOR_BIAS;\n", l))
  }
  if (prior_weights_dist == "horseshoe") {
    model_block <- paste0(model_block, "  w_out_raw ~ normal(0, 1);\n")
    model_block <- paste0(model_block, "  lambda_w_out ~ cauchy(0, 1);\n")
    model_block <- paste0(model_block, "  tau_w_out ~ cauchy(0, 1);\n")
  } else {
    if (out_act_fn == 3) {
      model_block <- paste0(model_block, "  to_vector(w_out) ~ PRIOR_WEIGHT;\n")
    } else {
      model_block <- paste0(model_block, "  w_out ~ PRIOR_WEIGHT;\n")
    }
  }
  model_block <- paste0(model_block, "  b_out ~ PRIOR_BIAS;\n")

  if (out_act_fn == 1) {
    model_block <- paste0(model_block, "  sigma ~ PRIOR_SIGMA;\n")
  }

  # Forward pass logic
  model_block <- paste0(model_block, "
  // Forward pass
  matrix[n, nodes[1]] z1 = X * w1 + rep_matrix(b1', n);
  matrix[n, nodes[1]] a1 = z1; // Placeholder
  ")

  # Activation function logic for first layer
  model_block <- paste0(model_block, "
  if (act_fn[1] == 1) a1 = tanh(z1);
  if (act_fn[1] == 2) a1 = inv_logit(z1);
  if (act_fn[1] == 3) a1 = log1p_exp(z1);
  if (act_fn[1] == 4) a1 = fmax(0, z1);
  // if act_fn[1] == 5, it remains linear (a1=z1)
  ")

  # Subsequent hidden layers
  if (num_layers > 1) {
    for (l in 2:num_layers) {
      model_block <- paste0(model_block, sprintf("
  matrix[n, nodes[%d]] z%d = a%d * w%d + rep_matrix(b%d', n);
  matrix[n, nodes[%d]] a%d = z%d; // Placeholder
      ", l, l, l - 1, l, l, l, l, l))
      model_block <- paste0(model_block, sprintf("
  if (act_fn[%d] == 1) a%d = tanh(z%d);
  if (act_fn[%d] == 2) a%d = inv_logit(z%d);
  if (act_fn[%d] == 3) a%d = log1p_exp(z%d);
  if (act_fn[%d] == 4) a%d = fmax(0, z%d);
      ", l, l, l, l, l, l, l, l, l, l, l, l))
    }
  }

  # Output layer and likelihood
  last_a <- paste0("a", num_layers)
  if (out_act_fn == 3) { # Multiclass
    model_block <- paste0(model_block, sprintf("
  matrix[n, K] out_layer = %s * w_out + rep_matrix(b_out', n);
  for (i in 1:n) {
    y[i] ~ categorical_logit(out_layer[i]');
  }
", last_a))
  } else { # Regression or Binary
    model_block <- paste0(model_block, sprintf("
  vector[n] out_layer = %s * w_out + rep_vector(b_out[1], n);
", last_a))
    if (out_act_fn == 1) { # Regression
      model_block <- paste0(model_block, "  y ~ normal(out_layer, sigma);\n")
    } else { # Binary
      model_block <- paste0(model_block, "  y ~ bernoulli_logit(out_layer);\n")
    }
  }
  model_block <- paste0(model_block, "}\n")

  # --- Generated Quantities Block ---
  gq_block <- "
generated quantities {
  vector[n] log_lik;
"
  gq_block <- paste0(gq_block, if (out_act_fn == 1) "  vector[n] y_rep;\n" else "  array[n] int y_rep;\n")
  gq_block <- paste0(gq_block, "
  {
    // Re-calculate forward pass to get final predictions
    matrix[n, nodes[1]] z1 = X * w1 + rep_matrix(b1', n);
    matrix[n, nodes[1]] a1 = z1;
    if (act_fn[1] == 1) a1 = tanh(z1);
    if (act_fn[1] == 2) a1 = inv_logit(z1);
    if (act_fn[1] == 3) a1 = log1p_exp(z1);
    if (act_fn[1] == 4) a1 = fmax(0, z1);
")

  if (num_layers > 1) {
    for (l in 2:num_layers) {
      gq_block <- paste0(gq_block, sprintf("
    matrix[n, nodes[%d]] z%d = a%d * w%d + rep_matrix(b%d', n);
    matrix[n, nodes[%d]] a%d = z%d;
    if (act_fn[%d] == 1) a%d = tanh(z%d);
    if (act_fn[%d] == 2) a%d = inv_logit(z%d);
    if (act_fn[%d] == 3) a%d = log1p_exp(z%d);
    if (act_fn[%d] == 4) a%d = fmax(0, z%d);
", l, l, l - 1, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l))
    }
  }

  if (out_act_fn == 3) { # Multiclass
    gq_block <- paste0(gq_block, sprintf("
    matrix[n, K] y_hat_mat = %s * w_out + rep_matrix(b_out', n);
    for (i in 1:n) {
      log_lik[i] = categorical_logit_lpmf(y[i] | y_hat_mat[i]');
      y_rep[i] = categorical_rng(softmax(y_hat_mat[i]'));
    }
", last_a))
  } else { # Regression or Binary
    gq_block <- paste0(gq_block, sprintf("
    vector[n] y_hat = %s * w_out + rep_vector(b_out[1], n);
", last_a))
    if (out_act_fn == 1) { # Regression
      gq_block <- paste0(gq_block, "
    for (i in 1:n) {
      log_lik[i] = normal_lpdf(y[i] | y_hat[i], sigma);
      y_rep[i] = normal_rng(y_hat[i], sigma);
    }
")
    } else { # Binary
      gq_block <- paste0(gq_block, "
    for (i in 1:n) {
      log_lik[i] = bernoulli_logit_lpmf(y[i] | y_hat[i]);
      y_rep[i] = bernoulli_rng(inv_logit(y_hat[i]));
    }
")
    }
  }
  gq_block <- paste0(gq_block, "  }\n}\n")

  # --- Combine Blocks ---
  full_model_code <- paste0(data_block, params_block, tparams_block, model_block, gq_block)

  return(full_model_code)
}

#' Generate Stan code for regression
#' @keywords internal
#' @noRd
generate_stan_code_cont <- function(num_layers, nodes) {
  if (length(nodes) != num_layers) {
    stop("The length of 'nodes' must match the number of hidden layers 'num_layers'.")
  }
  if (any(nodes < 1)) {
    stop("The number of nodes in each hidden layer must be at least 1.")
  }
  generate_stan_code(num_layers = num_layers, nodes = nodes, out_act_fn = 1)
}

#' Generate Stan code for binary classification
#' @keywords internal
#' @noRd
generate_stan_code_bin <- function(num_layers, nodes) {
  if (length(nodes) != num_layers) {
    stop("The length of 'nodes' must match the number of hidden layers 'num_layers'.")
  }
  if (any(nodes < 1)) {
    stop("The number of nodes in each hidden layer must be at least 1.")
  }
  generate_stan_code(num_layers = num_layers, nodes = nodes, out_act_fn = 2)
}

#' Generate Stan code for multiclass classification
#' @keywords internal
#' @noRd
generate_stan_code_cat <- function(num_layers, nodes) {
  if (length(nodes) != num_layers) {
    stop("The length of 'nodes' must match the number of hidden layers 'num_layers'.")
  }
  if (any(nodes < 1)) {
    stop("The number of nodes in each hidden layer must be at least 1.")
  }
  generate_stan_code(num_layers = num_layers, nodes = nodes, out_act_fn = 3)
}