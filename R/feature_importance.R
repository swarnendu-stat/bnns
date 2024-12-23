feature_importance <- function(object, ...){
  w1 <- rstan::extract(object$fit)$w1

}
