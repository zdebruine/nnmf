#' Non-negative Matrix Factorization by Alternating Least Squares
#'
#' @description
#' This is the classical NMF algorithm implemented in-core that requires transposition of the input data.
#'
#' @details
#' Factors in W and H are normalized to their 2-norm after each iteration.
#'
#' The input matrix will be copied to float type on the C++ back-end. This requires additional RAM, but provides significant speedups.
#'
#' Training set reconstruction error is not computed during model fitting. If it is desired, runtime may increase significantly.
#'
#' Masking the test set significantly decreases scalability of the algorithm, which will be noticeable at moderate ranks (50-200) and will begin to render models computationally intractable (appx. k > 500)
#'
#' By default, W is initialized with random normal values. H is first solved given the input data and W, and thus is not initialized. You may provide a custom initialization for W if desired, but NNDSVD is not recommended due to the local minimum it already supplies that makes it difficult for NMF to discover a truly optimal solution.
#'
#' @param data Dense or sparse matrix
#' @param k rank
#' @param test_size Integer giving the proportion of values that will be reserved for the test set in a random speckled pattern. This proportion will be inverted and rounded to the nearest integer for efficient RNG purposes.
#' @param test_seed Random seed determining the test set.
#' @param seed A number giving the random seed for initializing W, or a matrix giving initial W of dimensions m x k
#' @param tol stopping criterion giving the relative 2-norm distance between W across consecutive iterations at which to call convergence
#' @param epochs maximum number of alternating least squares updates for which to fit
#' @param verbose Print logging output to console
#' @param L1 L1 penalty, optionally a vector of two giving penalty on c(W, H) individually
#' @param L2 L2 penalty, optionally a vector of two giving penalty on c(W, H) individually
#' @param ortho orthogonality penalty, optionally a vector of two giving penalty on c(W, H) individually
#' @param log Parameters that are nearly free to compute will be automatically logged. In addition, you may specify any of c("test_loss", "train_loss")
#' @param num_threads Number of threads to use for parallelization. If 0, will use all available threads.
#' 
#' @import RcppEigen
#' @export
#' @useDynLib nnmf, .registration = TRUE
#'
als_nnmf <- function(data, k, test_size = 0.0615, test_seed = 129, seed = 42, tol = 1e-4, epochs = 100, verbose = TRUE, L1 = c(0, 0), L2 = c(0, 0), ortho = c(0, 0), log = NULL, num_threads = 0, fast_nnls = FALSE) {

      inv_test_density <- pmax(2, round(1 / test_size))
      if(is.matrix(seed)){
            if (ncol(seed) != k || nrow(seed) != nrow(data)) {
              stop("Seed matrix must have dimensions m x k, where m is the number of rows in data and k is the rank.")
            }
            w_init <- seed
      } else {
            if (length(seed) != 1 || !is.numeric(seed)) {
              stop("Seed must be a single numeric value or a matrix of dimensions m x k.")
            }
            set.seed(as.integer(seed))
            w_init <- matrix(runif(nrow(data) * k), nrow = nrow(data), ncol = k)
            w_init <- w_init / sqrt(rowSums(w_init^2))
      }

      if(length(L1) == 1) L1 <- rep(L1, 2)
      if(length(L2) == 1) L2 <- rep(L2, 2)
      if(length(ortho) == 1) ortho <- rep(ortho, 2)
      if(length(L1) != 2 || length(L2) != 2 || length(ortho) != 2) {
            stop("L1, L2, and ortho must be either a single value or a vector of two values.")
      }

      cpp_als_nnmf(data, k, inv_test_density, test_seed, t(w_init), tol, epochs, verbose, L1, L2, ortho, "train_loss" %in% log, "test_loss" %in% log, num_threads)
}
