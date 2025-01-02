#' Generate Stan Code Based on Output Activation Function
#'
#' This function serves as a wrapper to generate Stan code for Bayesian neural networks
#' tailored to different types of response variables. Based on the specified output
#' activation function (`out_act_fn`), it delegates the code generation to the
#' appropriate function for continuous, binary, or categorical response models.
#'
#' @param num_layers An integer specifying the number of hidden layers in the neural network.
#' @param nodes A vector of integers, where each element specifies the number of nodes
#'   in the corresponding hidden layer. The length of the vector must match `num_layers`.
#' @param out_act_fn An integer specifying the output activation function, determining
#'   the type of response variable. Supported values are:
#'   - `1`: Continuous response (identity function as output layer).
#'   - `2`: Binary response (sigmoid function as output layer).
#'   - `3`: Categorical response (softmax function as output layer).
#'
#' @return A character string containing the Stan code for the specified Bayesian neural network model.
#'   The Stan model includes data, parameters, transformed parameters, and model blocks,
#'   adjusted based on the specified response type.
#'
#' @details
#' This function dynamically calls one of the following functions based on the value of `out_act_fn`:
#' - **Continuous response:** Calls `generate_stan_code_cont`.
#' - **Binary response:** Calls `generate_stan_code_bin`.
#' - **Categorical response:** Calls `generate_stan_code_cat`.
#'
#' If an unsupported value is provided for `out_act_fn`, the function throws an error.
#' The generated Stan code is adapted for the response type, including appropriate
#' likelihood functions and transformations.
#'
#' @examples
#' # Generate Stan code for a continuous response model
#' stan_code <- generate_stan_code(num_layers = 2, nodes = c(10, 5), out_act_fn = 1)
#' cat(stan_code)
#'
#' # Generate Stan code for a binary response model
#' stan_code <- generate_stan_code(num_layers = 2, nodes = c(10, 5), out_act_fn = 2)
#' cat(stan_code)
#'
#' # Generate Stan code for a categorical response model
#' stan_code <- generate_stan_code(num_layers = 2, nodes = c(10, 5), out_act_fn = 3)
#' cat(stan_code)
#'
#' @seealso
#' - [generate_stan_code_cont]: For continuous response models.
#' - [generate_stan_code_bin]: For binary response models.
#' - [generate_stan_code_cat]: For categorical response models.
#'
#' @export


generate_stan_code <- function(num_layers, nodes, out_act_fn = 1) {
  if (out_act_fn == 1) {
    return(generate_stan_code_cont(num_layers = num_layers, nodes = nodes))
  } else if (out_act_fn == 2) {
    return(generate_stan_code_bin(num_layers = num_layers, nodes = nodes))
  } else if (out_act_fn == 3) {
    return(generate_stan_code_cat(num_layers = num_layers, nodes = nodes))
  } else {
    stop("This output activation layer is not currently supported")
  }
}

#' Generate Stan Code for Continuous Response Models
#'
#' This function generates Stan code for a Bayesian neural network model
#' designed to predict continuous response variables. The Stan code is dynamically
#' constructed based on the specified number of hidden layers and nodes per layer.
#' It supports various activation functions for the hidden layers, including
#' tanh, sigmoid, softplus and relu.
#'
#' @param num_layers An integer specifying the number of hidden layers in the neural network.
#' @param nodes A vector of integers, where each element specifies the number of nodes
#'   in the corresponding hidden layer. The length of the vector must match `num_layers`.
#'
#' @return A character string containing the Stan code for the specified Bayesian neural network model.
#'   The Stan model includes data, parameters, transformed parameters, and model blocks.
#'   The code is adjusted based on whether the network has one or multiple hidden layers.
#'
#' @details
#' The generated Stan code models a continuous response variable using a neural network.
#' The hidden layers apply the specified activation functions, while the output layer
#' performs a linear transformation to predict the response. The likelihood assumes
#' normally distributed residuals.
#'
#' - **For one hidden layer:** The function simplifies the Stan code structure.
#' - **For multiple hidden layers:** The code dynamically includes additional layers
#'   based on the input arguments.
#'
#' Supported activation functions for the hidden layers:
#' - 1: Tanh
#' - 2: Sigmoid
#' - 3: Softplus
#' - 4: ReLU
#'
#' @examples
#' # Generate Stan code for a single hidden layer with 10 nodes
#' stan_code <- generate_stan_code_cont(1, c(10))
#' cat(stan_code)
#'
#' # Generate Stan code for two hidden layers with 8 and 4 nodes
#' stan_code <- generate_stan_code_cont(2, c(8, 4))
#' cat(stan_code)
#'
#' @export

generate_stan_code_cont <- function(num_layers, nodes) {
  if (length(nodes) != num_layers || any(nodes <= 0)) {
    stop("Ensure 'nodes' length matches 'num_layers' and all values are positive")
  }

  if (num_layers == 1) {
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
  to_vector(w1) ~ PRIOR_WEIGHT;
  b1 ~ PRIOR_BIAS;
  w_out ~ PRIOR_WEIGHT;
  b_out ~ PRIOR_BIAS;
  sigma ~ PRIOR_SIGMA;
  y ~ normal(y_hat, sigma);
}
")
  } else {
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
      params <- c(
        params,
        paste0("  matrix[nodes[", l - 1, "], nodes[", l, "]] w", l, ";"),
        paste0("  vector[nodes[", l, "]] b", l, ";")
      )
    }
    params <- c(params, "  vector[nodes[L]] w_out;", "  real b_out;", "  real<lower=0> sigma;", "}")
    sections$params <- paste(params, collapse = "\n")

    # Transformed parameters block
    transformed <- c("transformed parameters {", "  matrix[n, nodes[1]] z1;", "  matrix[n, nodes[1]] a1;")

    for (l in seq(2, num_layers)) {
      transformed <- c(
        transformed,
        paste0("  matrix[n, nodes[", l, "]] z", l, ";"),
        paste0("  matrix[n, nodes[", l, "]] a", l, ";")
      )
    }

    transformed <- c(transformed, "  vector[n] y_hat;", "  z1 = X * w1 + rep_matrix(b1', n);")
    for (l in seq(1, num_layers)) {
      transformed <- c(
        transformed,
        paste0("  if (act_fn[", l, "] == 1) a", l, " = tanh(z", l, ");"),
        paste0("  else if (act_fn[", l, "] == 2) a", l, " = inv_logit(z", l, ");"),
        paste0("  else if (act_fn[", l, "] == 3) a", l, " = log(1 + exp(z", l, "));"),
        paste0("  else a", l, " = fmax(rep_matrix(0, n, nodes[", l, "]), z", l, ");")
      )
      if (l < num_layers) {
        transformed <- c(
          transformed,
          paste0("  z", l + 1, " = a", l, " * w", l + 1, " + rep_matrix(b", l + 1, "', n);")
        )
      }
    }
    transformed <- c(transformed, paste0("  y_hat = a", num_layers, " * w_out + b_out;"), "}")
    sections$transformed <- paste(transformed, collapse = "\n")

    # Model block
    model <- c("model {", "  to_vector(w1) ~ PRIOR_WEIGHT;", "  b1 ~ PRIOR_BIAS;")
    for (l in seq(2, num_layers)) {
      model <- c(
        model,
        paste0("  to_vector(w", l, ") ~ PRIOR_WEIGHT;"),
        paste0("  b", l, " ~ PRIOR_BIAS;")
      )
    }

    model <- c(model, "  w_out ~ PRIOR_WEIGHT;", "  b_out ~ PRIOR_BIAS;", "  sigma ~ PRIOR_SIGMA;", "  y ~ normal(y_hat, sigma);", "}")
    sections$model <- paste(model, collapse = "\n")

    # Combine all sections
    stan_code <- paste(sections, collapse = "\n\n")
    return(stan_code)
  }
}

#' Generate Stan Code for Binary Response Models
#'
#' This function generates Stan code for a Bayesian neural network model
#' designed to predict binary response variables. The Stan code is dynamically
#' constructed based on the specified number of hidden layers and nodes per layer.
#' It supports various activation functions for the hidden layers, including
#' tanh, sigmoid, softplus and relu. The model uses a Bernoulli likelihood for binary outcomes.
#'
#' @param num_layers An integer specifying the number of hidden layers in the neural network.
#' @param nodes A vector of integers, where each element specifies the number of nodes
#'   in the corresponding hidden layer. The length of the vector must match `num_layers`.
#'
#' @return A character string containing the Stan code for the specified Bayesian neural network model.
#'   The Stan model includes data, parameters, transformed parameters, and model blocks.
#'   The code is adjusted based on whether the network has one or multiple hidden layers.
#'
#' @details
#' The generated Stan code models a binary response variable using a neural network.
#' The hidden layers apply the specified activation functions, while the output layer
#' applies the logistic function to predict the probability of the binary outcome.
#'
#' - **For one hidden layer:** The function simplifies the Stan code structure.
#' - **For multiple hidden layers:** The code dynamically includes additional layers
#'   based on the input arguments.
#'
#' Supported activation functions for the hidden layers:
#' - 1: Tanh
#' - 2: Sigmoid
#' - 3: Softplus
#' - 4: ReLU
#'
#' The output layer uses a logistic transformation (`inv_logit`) to constrain
#' predictions between 0 and 1, which aligns with the Bernoulli likelihood.
#'
#' @examples
#' # Generate Stan code for a single hidden layer with 10 nodes
#' stan_code <- generate_stan_code_bin(1, c(10))
#' cat(stan_code)
#'
#' # Generate Stan code for two hidden layers with 8 and 4 nodes
#' stan_code <- generate_stan_code_bin(2, c(8, 4))
#' cat(stan_code)
#'
#' @export

generate_stan_code_bin <- function(num_layers, nodes) {
  if (length(nodes) != num_layers || any(nodes <= 0)) {
    stop("Ensure 'nodes' length matches 'num_layers' and all values are positive")
  }

  if (num_layers == 1) {
    return("data {
  int<lower=1> n;
  int<lower=1> m;
  int<lower=1> L;
  int<lower=1> nodes;
  matrix[n, m] X;
  array[n] int<lower=0, upper=1> y;
  int<lower=1, upper=3> act_fn;
}

parameters {
  matrix[m, nodes] w1;
  vector[nodes] b1;
  vector[nodes] w_out;
  real b_out;
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
  to_vector(w1) ~ PRIOR_WEIGHT;
  b1 ~ PRIOR_BIAS;
  w_out ~ PRIOR_WEIGHT;
  b_out ~ PRIOR_BIAS;
  y ~ bernoulli_logit(y_hat);
}
")
  } else {
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
  array[n] int<lower=0, upper=1> y;
  int<lower=1> act_fn[L];
}"

    # Parameters block
    params <- c("parameters {", "  matrix[m, nodes[1]] w1;", "  vector[nodes[1]] b1;")
    for (l in seq(2, num_layers)) {
      params <- c(
        params,
        paste0("  matrix[nodes[", l - 1, "], nodes[", l, "]] w", l, ";"),
        paste0("  vector[nodes[", l, "]] b", l, ";")
      )
    }
    params <- c(params, "  vector[nodes[L]] w_out;", "  real b_out;", "}")
    sections$params <- paste(params, collapse = "\n")

    # Transformed parameters block
    transformed <- c("transformed parameters {", "  matrix[n, nodes[1]] z1;", "  matrix[n, nodes[1]] a1;")

    for (l in seq(2, num_layers)) {
      transformed <- c(
        transformed,
        paste0("  matrix[n, nodes[", l, "]] z", l, ";"),
        paste0("  matrix[n, nodes[", l, "]] a", l, ";")
      )
    }

    transformed <- c(transformed, "  vector[n] y_hat;", "  z1 = X * w1 + rep_matrix(b1', n);")
    for (l in seq(1, num_layers)) {
      transformed <- c(
        transformed,
        paste0("  if (act_fn[", l, "] == 1) a", l, " = tanh(z", l, ");"),
        paste0("  else if (act_fn[", l, "] == 2) a", l, " = inv_logit(z", l, ");"),
        paste0("  else if (act_fn[", l, "] == 3) a", l, " = log(1 + exp(z", l, "));"),
        paste0("  else a", l, " = fmax(rep_matrix(0, n, nodes[", l, "]), z", l, ");")
      )
      if (l < num_layers) {
        transformed <- c(
          transformed,
          paste0("  z", l + 1, " = a", l, " * w", l + 1, " + rep_matrix(b", l + 1, "', n);")
        )
      }
    }
    transformed <- c(transformed, paste0("  y_hat = a", num_layers, " * w_out + b_out;"), "}")
    sections$transformed <- paste(transformed, collapse = "\n")

    # Model block
    model <- c("model {", "  to_vector(w1) ~ PRIOR_WEIGHT;", "  b1 ~ PRIOR_BIAS;")
    for (l in seq(2, num_layers)) {
      model <- c(
        model,
        paste0("  to_vector(w", l, ") ~ PRIOR_WEIGHT;"),
        paste0("  b", l, " ~ PRIOR_BIAS;")
      )
    }

    model <- c(model, "  w_out ~ PRIOR_WEIGHT;", "  b_out ~ PRIOR_BIAS;", "  y ~ bernoulli_logit(y_hat);", "}")
    sections$model <- paste(model, collapse = "\n")

    # Combine all sections
    stan_code <- paste(sections, collapse = "\n\n")
    return(stan_code)
  }
}

#' Generate Stan Code for Neural Networks with Categorical Response
#'
#' This function generates Stan code for modeling a categorical response using
#' neural networks with multiple layers. The generated code supports customizable
#' activation functions for each layer and softmax-based prediction for the categorical output.
#'
#' @param num_layers Integer. Number of layers in the neural network.
#' @param nodes Integer vector. Number of nodes in each layer. The length of
#'   this vector must match `num_layers`, and all values must be positive.
#'
#' @return A string containing the Stan code for the specified neural network
#'   architecture and categorical response model.
#'
#' @details
#' The Stan code includes the following components:
#' - **Data Block**: Defines inputs, response variable, layer configurations, and activation functions.
#' - **Parameters Block**: Declares weights and biases for all layers and the output layer.
#' - **Transformed Parameters Block**: Computes intermediate outputs (`z` and `a`) for each layer
#'   and calculates the final predictions (`y_hat`) using the softmax function.
#' - **Model Block**: Specifies priors for parameters and models the categorical response
#'   using `categorical_logit`.
#'
#' Supported activation functions for each layer:
#' - 1: Hyperbolic tangent (`tanh`)
#' - 2: Logistic sigmoid (`inv_logit`)
#' - 3: Softplus (`log(1 + exp(x))`)
#' - Default: Rectified linear unit (ReLU)
#'
#' The categorical response (`y`) is assumed to take integer values from 1 to `K`,
#' where `K` is the total number of categories.
#'
#' @examples
#' # Generate Stan code for a neural network with 3 layers
#' num_layers <- 3
#' nodes <- c(10, 8, 6) # 10 nodes in the first layer, 8 in the second, 6 in the third
#' stan_code <- generate_stan_code_cat(num_layers, nodes)
#' cat(stan_code)
#'
#' @seealso [generate_stan_code_bin()], [generate_stan_code_cont()]
#'
#' @export

generate_stan_code_cat <- function(num_layers, nodes) {
  if (length(nodes) != num_layers || any(nodes <= 0)) {
    stop("Ensure 'nodes' length matches 'num_layers' and all values are positive")
  }

  if (num_layers == 1) {
    return("data {
  int<lower=1> n;
  int<lower=1> m;
  int<lower=1> L;
  int<lower=1> nodes;
  matrix[n, m] X;
  array[n] int<lower=1> y;
  int<lower=2> K; // Number of categories
  int<lower=1, upper=3> act_fn;
}

parameters {
  matrix[m, nodes] w1;
  vector[nodes] b1;
  matrix[nodes, K] w_out;
  vector[K] b_out;
}

transformed parameters {
  matrix[n, nodes] z1;
  matrix[n, nodes] a1;
  matrix[n, K] y_hat;

  z1 = X * w1 + rep_matrix(b1', n);
  if (act_fn == 1) a1 = tanh(z1);
  else if (act_fn == 2) a1 = inv_logit(z1);
  else if (act_fn == 3) a1 = log(1 + exp(z1));
  else a1 = fmax(rep_matrix(0, n, nodes), z1);

  y_hat = a1 * w_out + rep_matrix(b_out', n);
}

model {
  to_vector(w1) ~ PRIOR_WEIGHT;
  b1 ~ PRIOR_BIAS;
  to_vector(w_out) ~ PRIOR_WEIGHT;
  b_out ~ PRIOR_BIAS;
  for (i in 1:n) y[i] ~ categorical_logit(y_hat[i]');
}
")
  } else {
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
  array[n] int<lower=1> y;
  int<lower=1> act_fn[L];
  int<lower=2> K; // Number of categories
}"

    # Parameters block
    params <- c("parameters {", "  matrix[m, nodes[1]] w1;", "  vector[nodes[1]] b1;")
    for (l in seq(2, num_layers)) {
      params <- c(
        params,
        paste0("  matrix[nodes[", l - 1, "], nodes[", l, "]] w", l, ";"),
        paste0("  vector[nodes[", l, "]] b", l, ";")
      )
    }
    params <- c(params, "  matrix[nodes[L], K] w_out;", "  vector[K] b_out;", "}")
    sections$params <- paste(params, collapse = "\n")

    # Transformed parameters block
    transformed <- c("transformed parameters {", "  matrix[n, nodes[1]] z1;", "  matrix[n, nodes[1]] a1;")

    for (l in seq(2, num_layers)) {
      transformed <- c(
        transformed,
        paste0("  matrix[n, nodes[", l, "]] z", l, ";"),
        paste0("  matrix[n, nodes[", l, "]] a", l, ";")
      )
    }

    transformed <- c(transformed, "  matrix[n, K] y_hat;", "  z1 = X * w1 + rep_matrix(b1', n);")
    for (l in seq(1, num_layers)) {
      transformed <- c(
        transformed,
        paste0("  if (act_fn[", l, "] == 1) a", l, " = tanh(z", l, ");"),
        paste0("  else if (act_fn[", l, "] == 2) a", l, " = inv_logit(z", l, ");"),
        paste0("  else if (act_fn[", l, "] == 3) a", l, " = log(1 + exp(z", l, "));"),
        paste0("  else a", l, " = fmax(rep_matrix(0, n, nodes[", l, "]), z", l, ");")
      )
      if (l < num_layers) {
        transformed <- c(
          transformed,
          paste0("  z", l + 1, " = a", l, " * w", l + 1, " + rep_matrix(b", l + 1, "', n);")
        )
      }
    }
    transformed <- c(transformed, paste0("  y_hat = a", num_layers, " * w_out + rep_matrix(b_out', n);"), "}")
    sections$transformed <- paste(transformed, collapse = "\n")

    # Model block
    model <- c("model {", "  to_vector(w1) ~ PRIOR_WEIGHT;", "  b1 ~ PRIOR_BIAS;")
    for (l in seq(2, num_layers)) {
      model <- c(
        model,
        paste0("  to_vector(w", l, ") ~ PRIOR_WEIGHT;"),
        paste0("  b", l, " ~ PRIOR_BIAS;")
      )
    }

    model <- c(
      model, "  to_vector(w_out) ~ PRIOR_WEIGHT;", "  b_out ~ PRIOR_BIAS;",
      "  for (i in 1:n) y[i] ~ categorical_logit(y_hat[i]');", "}"
    )
    sections$model <- paste(model, collapse = "\n")

    # Combine all sections
    stan_code <- paste(sections, collapse = "\n\n")
    return(stan_code)
  }
}
