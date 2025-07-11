#' Non-negative Matrix Factorization by Alternating Least Squares
#'
#' @description
#' This is the classical NMF algorithm implemented in-core that requires transposition of the input data.
#'
#' @details
#' Factors in W and H are normalized to their 2-norm after each iteration, and the resulting scaling factor is given in the scaling diagonal.
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
#' @param inv_test_density Integer giving the inverse density of a random speckled test set (e.g. 16 = 6.125% of data)
#' @param seed A number giving the random seed, or a matrix used to initialize W of dimensions m x k
#' @param tol stopping criterion giving the relative 2-norm distance between W across consecutive iterations at which to call convergence
#' @param epochs maximum number of alternating least squares updates for which to fit
#' @param verbose Print logging output to console
#' @param L1 L1 penalty, optionally a vector of two giving penalty on c(W, H) individually
#' @param L2 L2 penalty, optionally a vector of two giving penalty on c(W, H) individually
#' @param ortho orthogonality penalty, optionally a vector of two giving penalty on c(W, H) individually
#' @param logger Parameters that are nearly free to compute will be automatically logged. In addition, you may specify any of c("test_loss", "train_loss")
#'
#' @import RcppEigen
#'
als_nnmf <- function(data, k, inv_test_density = 16, seed = 42, tol = 1e-4, epochs = 100, verbose = TRUE, L1 = c(0, 0), L2 = c(0, 0), ortho = c(0, 0), logger = NULL) {
  xxt(data)
}
