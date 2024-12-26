#' Measure Performance for Continuous Response Models
#'
#' Evaluates the performance of a continuous response model using RMSE and MAE.
#'
#' @param obs A numeric vector of observed (true) values.
#' @param pred A numeric vector of predicted values.
#'
#' @return A list containing:
#' \describe{
#'   \item{`rmse`}{Root Mean Squared Error.}
#'   \item{`mae`}{Mean Absolute Error.}
#' }
#'
#' @examples
#' \dontrun{
#' obs <- c(3.2, 4.1, 5.6)
#' pred <- c(3.0, 4.3, 5.5)
#' measure_cont(obs, pred)
#' # Returns: list(rmse = 0.1732051, mae = 0.1666667)
#' }
#'
#' @export
measure_cont <- function(obs, pred) {
  if(!is.null(dim(pred))){
    if(length(dim(pred)) == 2){
      pred <- rowMeans(pred)
    }
  }
  return(list(rmse = sqrt(mean((obs - pred)^2)),
              mae = mean(abs(obs - pred))))
}

#' Measure Performance for Binary Classification Models
#'
#' Evaluates the performance of a binary classification model using a confusion matrix and accuracy.
#'
#' @param obs A numeric or integer vector of observed binary class labels (0 or 1).
#' @param pred A numeric vector of predicted probabilities for the positive class.
#' @param cut A numeric threshold (between 0 and 1) to classify predictions into binary labels.
#'
#' @return A list containing:
#' \describe{
#'   \item{`conf_mat`}{A confusion matrix comparing observed and predicted class labels.}
#'   \item{`accuracy`}{The proportion of correct predictions.}
#'   \item{`ROC`}{ROC generated using `pROC::roc`}
#'   \item{`AUC`}{Area under the ROC curve.}
#' }
#'
#' @examples
#' \dontrun{
#' obs <- c(1, 0, 1, 1, 0)
#' pred <- c(0.9, 0.4, 0.8, 0.7, 0.3)
#' cut <- 0.5
#' measure_bin(obs, pred, cut)
#' # Returns: list(conf_mat = <confusion matrix>, accuracy = 1, ROC = <ROC>, AUC = 1)
#' }
#'
#' @export
measure_bin <- function(obs, pred, cut = 0.5) {
  if(!is.null(dim(pred))){
    if(length(dim(pred)) == 2){
      pred <- rowMeans(pred)
    }
  }
  pred_label <- ifelse(pred >= cut, 1, 0)
  conf_mat <- table(obs, pred_label)
  ROC <- pROC::roc(response = obs, predictor = pred)
  return(list(conf_mat = conf_mat,
              accuracy = sum(diag(conf_mat)) / sum(conf_mat),
              ROC = ROC,
              AUC = as.numeric(ROC$auc)))
}

#' Measure Performance for Multi-Class Classification Models
#'
#' Evaluates the performance of a multi-class classification model using log loss and multiclass AUC.
#'
#' @param obs A factor vector of observed class labels. Each level represents a unique class.
#' @param pred A numeric matrix of predicted probabilities, where each row corresponds to an observation,
#' and each column corresponds to a class. The number of columns must match the number of levels in `obs`.
#'
#' @return A list containing:
#' \describe{
#'   \item{`log_loss`}{The negative log-likelihood averaged across observations.}
#'   \item{`ROC`}{ROC generated using `pROC::roc`}
#'   \item{`AUC`}{The multiclass Area Under the Curve (AUC) as computed by `pROC::multiclass.roc`.}
#' }
#'
#' @details
#' The log loss is calculated as:
#' \deqn{-\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(p_{ic})}
#' where \eqn{y_{ic}} is 1 if observation \eqn{i} belongs to class \eqn{c}, and \eqn{p_{ic}} is the
#' predicted probability for that class.
#'
#' The AUC is computed using the `pROC::multiclass.roc` function, which provides an overall measure
#' of model performance for multiclass classification.
#'
#' @examples
#' \dontrun{
#' library(pROC)
#' obs <- factor(c("A", "B", "C"), levels = LETTERS[1:3])
#' pred <- matrix(c(0.8, 0.1, 0.1,
#'                  0.2, 0.6, 0.2,
#'                  0.7, 0.2, 0.1),
#'                nrow = 3, byrow = TRUE)
#' measure_cat(obs, pred)
#' # Returns: list(log_loss = 1.012185, ROC = <ROC>, AUC = 0.75)
#' }
#'
#' @export
measure_cat <- function(obs, pred) {
  stopifnot("obs must be factor" = is.factor(obs))
  if(length(dim(pred)) == 3){
    pred <- t(apply(pred, 1, colMeans))
  }
  ROC <- pROC::multiclass.roc(obs, data.frame(pred) |> `colnames<-`(levels(obs)))
  obs <- as.numeric(obs)
  log_loss <- 0
  for (i in seq_along(obs)) {
    for (j in 1:ncol(pred)) {
      log_loss <- log_loss + ifelse(obs[i] == j, 1, 0) * log(pred[i, j])
    }
  }
  log_loss <- -log_loss / length(obs)

  return(list(log_loss = log_loss, ROC = ROC, AUC = as.numeric(ROC$auc)))
}
