#' Alternating Least Squares NMF
#'
#' @param data numeric matrix
#' @param k rank
#' @param tol convergence tolerance
#' @param epochs maximum iterations
#' @param verbose logical; print progress
#' @param log_train_loss logical; record training loss
#' @return list with factors and training log
#' @examples
#' mat <- matrix(abs(rnorm(20)), nrow = 5)
#' als_nmf(mat, k = 2)
#' @export
als_nmf <- function(data, k, tol = 1e-4, epochs = 100, verbose = TRUE,
                    log_train_loss = FALSE) {
  data <- as.matrix(data)
  .Call(`_nnmf_als_nmf`, data, as.integer(k), tol, as.integer(epochs),
        verbose, log_train_loss)
}
