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
#' Masking is exhaustively supported for any combination of sparse or dense input, masking zeros, masking a random speckled test set, and masking a user-supplied dense or sparse mask matrix. Values that are NA in the input data will be added to the mask matrix.
#' 
#' Masking significantly reduces scalability and performance of the algorithm, especially as rank increases. Optimizations are applied to the backend to handle various combinations of masks efficiently and apply the mask in a memory-efficient manner based on the sparsity of the mask.
#'
#' Custom initializations of W: By default, W is initialized with random normal values. H is first solved given the input data and W, and thus is not initialized. You may provide a custom initialization for W if desired, but NNDSVD is not recommended due to the local minimum it already supplies that makes it difficult for NMF to discover a truly optimal solution.
#'
#' Use set.seed() to control the test set mask and W initialization, unless you are providing a custom initialization for W, in which case set.seed() only controls the test set mask.
#' 
#' NA values in the data will automatically be masked. Be mindful of the number of NA values in your data, as this will affect the performance of the algorithm. If more than 10% of values are NA, a dense matrix will be used to store the indices of NA values, which may increase memory usage.
#'
#' @param data Dense or sparse matrix
#' @param k rank, or initialization for W of dimensions m x k where m is the number of rows in data and k is the rank.
#' @param test_size Integer giving the proportion of values that will be reserved for the test set in a random speckled pattern. This proportion will be inverted and rounded to the nearest integer for efficient RNG purposes.
#' @param tol stopping criterion giving the relative 2-norm distance between W across consecutive iterations at which to call convergence
#' @param epochs maximum number of alternating least squares updates for which to fit
#' @param verbose Print logging output to console
#' @param L1 L1 penalty on W and H, optionally a vector of two giving penalty on c(W, H) individually
#' @param L2 L2 penalty on W and H, optionally a vector of two giving penalty on c(W, H) individually
#' @param ortho orthogonality penalty on W and H, optionally a vector of two giving penalty on c(W, H) individually
#' @param log Character vector specifying which metrics to log each epoch. Options are any of c("ortho_loss", "sparsity", "train_loss", "test_loss"). Parameters that are nearly free to compute will be automatically logged. If train loss is logged and a test set exists, test loss is also logged. Both train and test loss exclude values that are otherwise masked (e.g. mask zeros, NA values, or custom mask matrix).
#' @param num_threads Number of threads to use for parallelization. If 0, will use all available threads.
#' @param mask A matrix (sparse or dense) with the same dimensions as input data, where 1 indicates a value that should be masked (not used in training) and 0 indicates a value that should be used in training. If NULL, no masking will be applied.
#' @param mask_zeros mask zeros in the input data, which will be treated as missing values. This is useful for datasets where zeros are not meaningful and should not be used in training.
#' 
#' @return A list containing:
#'   \item{w}{The W matrix (basis vectors)}
#'   \item{d}{A vector of scaling factors}
#'   \item{h}{The H matrix (coefficients)}
#'   \item{log}{A data.frame containing the logged parameters and metrics}
#'   \item{params}{A data.frame containing the input parameters}
#'
#' @import RcppEigen
#' @export
#' @useDynLib RcppML, .registration = TRUE
#'
nmf <- function(data, k, test_size = 0, tol = 1e-4, epochs = 100, verbose = TRUE, L1 = c(0, 0), L2 = c(0, 0), ortho = c(0, 0), log = NULL, num_threads = 0, mask = NULL, mask_zeros = FALSE) {
      # check if data contains any NA values
      test_seed <- sample(123:1e5, 1)
      mask_na <- NULL
      mask_class <- "dense"
      if (any(is.na(data))) {
        # if fewer than 10% of values are NA, we use a dgCMatrix to store the indices of NA values
        if(sum(is.na(data)) / prod(dim(data)) < 0.1) {
          if(class(data) == "dgCMatrix") {
            mask_na <- subset_dgCMatrix_by_NA(data)
          } else if(class(data) == "matrix") {
            mask_na <- as(as.matrix(is.na(data)), "dgCMatrix")
          }
        } else {
          if(class(data) == "dgCMatrix") {
            mask_na <- as.matrix(subset_dgCMatrix_by_NA(data))
          } else if(class(data) == "matrix") {
            mask_na <- matrix(0, nrow = nrow(data), ncol = ncol(data))
            mask_na[is.na(data)] <- 1
          }
        }
      }

      if(!is.null(mask)){
        # check that mask and input data have the same dimensions
        if(dim(mask) != dim(data)) {
          stop("Mask (if supplied) and input data must have the same dimensions.")
        }
        if(!is.null(mask_na)){
          # add mask to mask_na
          if(class(mask) == "matrix" || class(mask_na) == "matrix") {
            mask <- as.matrix(mask)
            mask_na <- as.matrix(mask)
            mask <- pmax(mask + mask_na, 1)
          } else {
            mask <- mask + mask_na
            mask@x <- pmax(mask@x, 1)
            mask_class <- "sparse"
          }
        }
      } else if(!is.null(mask_na)){
        mask <- mask_na
        if(class(mask) == "dgCMatrix") {
          mask_class <- "sparse"
        }
      } else {
        mask <- new("matrix")
      }

      if(test_size != 0){
        inv_test_density <- pmax(2, round(1 / test_size))
      } else {
        inv_test_density <- 0
      }

      if(is.matrix(k)){
            if (nrow(k) != nrow(data)) {
              stop("Initial W matrix (specified via k) must have dimensions m x k, where m is the number of rows in data and k is the rank.")
            }
            if(ncol(k) > nrow(data)) {
              stop("Initial W matrix (specified via k) must have at most m columns, where m is the number of rows in data.")
            }
            w_init <- k
            k <- ncol(w_init)
      } else {
            if (length(k) != 1 || !is.numeric(k)) {
              stop("k must be a single integer value or a matrix of dimensions m x k giving initial W.")
            }
            if (k <= 0 || k > nrow(data)) {
              stop("k must be a positive integer less than or equal to the number of rows in data.")
            }
            w_init <- matrix(runif(nrow(data) * k), nrow = nrow(data), ncol = k)
            w_init <- w_init / sqrt(rowSums(w_init^2))
      }

      if(length(L1) == 1) L1 <- rep(L1, 2)
      if(length(L2) == 1) L2 <- rep(L2, 2)
      if(length(ortho) == 1) ortho <- rep(ortho, 2)
      if(length(L1) != 2 || length(L2) != 2 || length(ortho) != 2) {
            stop("L1, L2, and ortho must be either a single value or a vector of two values where the first value gives the penalty on W and the second value gives the penalty on H.")
      }

      if(class(data) == "dgCMatrix"){
        is_sparse <- TRUE
      } else {
        is_sparse <- FALSE
        data <- as.matrix(data)
      }

      log <- unique(log)
      log_ortho_loss <- "ortho_loss" %in% log
      log_train_loss <- "train_loss" %in% log
      log_test_loss <- "test_loss" %in% log
      log_sparsity <- "sparsity" %in% log
      start_time <- Sys.time()
      if (is_sparse) {
        if(mask_class == "sparse"){
          model <- cpp_als_nmf_sparse_sparsemask(data, k, inv_test_density, test_seed, t(w_init), tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros)
        } else {
          model <- cpp_als_nmf_sparse_densemask(data, k, inv_test_density, test_seed, t(w_init), tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros)
        }
      } else {
        if(mask_class == "sparse"){
          model <- cpp_als_nmf_dense_sparsemask(data, k, inv_test_density, test_seed, t(w_init), tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros)
        } else {
          model <- cpp_als_nmf_dense_densemask(data, k, inv_test_density, test_seed, t(w_init), tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros)
        }
      }
      model$runtime <- difftime(Sys.time(), start_time, units = "secs")

      # add back dimnames to W, H, and d
      colnames(model$w) <- rownames(model$h) <- names(model$d) <- paste0("nmf", 1:ncol(model$w))
      if (!is.null(rownames(data))) {
        rownames(model$w) <- rownames(data)
      } 
      if (!is.null(colnames(data))) {
        colnames(model$h) <- colnames(data)
      }

      return(model)
}

subset_dgCMatrix_by_NA <- function(mat) {
  stopifnot(inherits(mat, "dgCMatrix"))

  # Get all non-zero entries and their positions
  x_vals <- mat@x
  row_idx <- mat@i
  col_ptr <- mat@p
  n_cols <- ncol(mat)

  # Match indices in @x corresponding to the target_value
  match_idx <- which(is.na(x_vals))
  if (length(match_idx) == 0) {
    # Return an empty sparse matrix with same dims
    return(Matrix(0, nrow = nrow(mat), ncol = ncol(mat), sparse = TRUE))
  }

  # Preallocate new column pointers
  new_p <- integer(n_cols + 1)
  new_i <- integer(length(match_idx))

  # Track position in output arrays
  out_pos <- 1

  for (j in seq_len(n_cols)) {
    # Get start and end of current column in x/i arrays
    start <- col_ptr[j] + 1
    end <- col_ptr[j + 1]
    col_range <- start:end

    # Match entries in current column
    col_match <- intersect(col_range, match_idx)

    # Append matched row indices and values
    if (length(col_match) > 0) {
      idx_range <- out_pos:(out_pos + length(col_match) - 1)
      new_i[idx_range] <- row_idx[col_match]
      out_pos <- out_pos + length(col_match)
    }

    # Update column pointer
    new_p[j + 1] <- out_pos - 1
  }

  # Construct new sparse matrix
  new_mat <- new("dgCMatrix",
                 i = new_i,
                 p = new_p,
                 x = rep(1, length(new_i)),
                 Dim = mat@Dim,
                 Dimnames = mat@Dimnames)
  return(new_mat)
}