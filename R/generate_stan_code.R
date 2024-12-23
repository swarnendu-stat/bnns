#' Generate the required Stan code
#'
#' @param num_layers Number of hidden layers.
#' @param nodes A vector of number of nodes in each hidden layer.
#' @returns A string with the required Stan code for Bayesian neural network.
#' @examples
#' generate_stan_code(3, c(16, 4, 2))
#' generate_stan_code(2, c(4, 2))

generate_stan_code <- function(num_layers, nodes) {
  if (length(nodes) != num_layers || any(nodes <= 0)) {
    stop("Ensure 'nodes' length matches 'num_layers' and all values are positive")
  }

  if(num_layers == 1){
    return("data {
  int<lower=1> n;
  int<lower=1> m;
  int<lower=1> L;
  int<lower=1> nodes;
  matrix[n, m] X;
  vector[n] y;
  int<lower=1, upper=3> act_fn;
}

parameters {
  matrix[m, nodes] w1;
  vector[nodes] b1;
  vector[nodes] w_out;
  real b_out;
  real<lower=0> sigma;
}

transformed parameters {
  matrix[n, nodes] z1;
  matrix[n, nodes] a1;
  vector[n] y_hat;

  z1 = X * w1 + rep_matrix(b1', n);
  if (act_fn == 1) a1 = tanh(z1);
  else if (act_fn == 2) a1 = inv_logit(z1);
  else if (act_fn == 3) a1 = log(1 + exp(z1));
  else a1 = fmax(rep_matrix(0, n, nodes), z1);

  y_hat = a1 * w_out + b_out;
}

model {
  to_vector(w1) ~ normal(0, 1);
  b1 ~ normal(0, 1);
  w_out ~ normal(0, 1);
  b_out ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ normal(y_hat, sigma);
}
")
  }else{

    # Initialize sections
    sections <- list()

    # Data block
    sections$data <- "
data {
  int<lower=1> n;
  int<lower=1> m;
  int<lower=1> L;
  int<lower=1> nodes[L];
  matrix[n, m] X;
  vector[n] y;
  int<lower=1> act_fn[L];
}"

    # Parameters block
    params <- c("parameters {", "  matrix[m, nodes[1]] w1;", "  vector[nodes[1]] b1;")
    for (l in seq(2, num_layers)) {
      params <- c(params,
                  paste0("  matrix[nodes[", l-1, "], nodes[", l, "]] w", l, ";"),
                  paste0("  vector[nodes[", l, "]] b", l, ";"))
    }
    params <- c(params, "  vector[nodes[L]] w_out;", "  real b_out;", "  real<lower=0> sigma;", "}")
    sections$params <- paste(params, collapse = "\n")

    # Transformed parameters block
    transformed <- c("transformed parameters {", "  matrix[n, nodes[1]] z1;", "  matrix[n, nodes[1]] a1;")

    for (l in seq(2, num_layers)) {
      transformed <- c(transformed,
                       paste0("  matrix[n, nodes[", l, "]] z", l, ";"),
                       paste0("  matrix[n, nodes[", l, "]] a", l, ";"))
    }

    transformed <- c(transformed, "  vector[n] y_hat;", "  z1 = X * w1 + rep_matrix(b1', n);")
    for (l in seq(1, num_layers)) {
      transformed <- c(transformed,
                       paste0("  if (act_fn[", l, "] == 1) a", l, " = tanh(z", l, ");"),
                       paste0("  else if (act_fn[", l, "] == 2) a", l, " = inv_logit(z", l, ");"),
                       paste0("  else if (act_fn[", l, "] == 3) a", l, " = log(1 + exp(z", l, "));"),
                       paste0("  else a", l, " = fmax(rep_matrix(0, n, nodes[", l, "]), z", l, ");"))
      if (l < num_layers) {
        transformed <- c(transformed,
                         paste0("  z", l+1, " = a", l, " * w", l+1, " + rep_matrix(b", l+1, "', n);"))
      }
    }
    transformed <- c(transformed, paste0("  y_hat = a", num_layers, " * w_out + b_out;"), "}")
    sections$transformed <- paste(transformed, collapse = "\n")

    # Model block
    model <- c("model {", "  to_vector(w1) ~ normal(0, 1);", "  b1 ~ normal(0, 1);")
    for (l in seq(2, num_layers)) {
      model <- c(model,
                 paste0("  to_vector(w", l, ") ~ normal(0, 1);"),
                 paste0("  b", l, " ~ normal(0, 1);"))
    }

    model <- c(model, "  w_out ~ normal(0, 1);", "  b_out ~ normal(0, 1);", "  sigma ~ normal(0, 1);", "  y ~ normal(y_hat, sigma);", "}")
    sections$model <- paste(model, collapse = "\n")

    # Combine all sections
    stan_code <- paste(sections, collapse = "\n\n")
    return(stan_code)
  }
}
