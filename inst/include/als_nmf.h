#ifndef ALS_NMF_H
#define ALS_NMF_H

#include "nnls.h"

/**
 * @brief Core Alternating Least Squares (ALS) routine for Non-negative Matrix Factorization (NMF) with masking and regularization.
 *
 * Alternates between solving for W and H using NNLS, supporting dense/sparse matrices, regularization (L1, L2, orthogonality),
 * and masking (zero entries, random test set, user mask matrix). Logs convergence and loss metrics for each epoch.
 *
 * @tparam MatrixType         Eigen dense or sparse matrix type for V
 * @tparam MaskMatrixType     Eigen dense or sparse matrix type for mask
 * @tparam MaskZeroEntries    If true, mask out zero entries in V
 * @tparam MaskTestSet        If true, apply a random test set mask for cross-validation
 * @tparam MaskMaskMatrix     If true, apply a user-supplied mask matrix
 * @param V                   Data matrix (n x m)
 * @param k                   Number of factors (rank)
 * @param TestMatrix          RandomSparseBinaryMatrix for test set masking
 * @param epochs              Maximum number of ALS iterations
 * @param tol                 Convergence tolerance (relative change in W)
 * @param verbose             If true, print progress each epoch
 * @param L1_W                L1 regularization for W
 * @param L1_H                L1 regularization for H
 * @param L2_W                L2 regularization for W
 * @param L2_H                L2 regularization for H
 * @param ortho_W             Orthogonality penalty for W
 * @param ortho_H             Orthogonality penalty for H
 * @param log_ortho_loss      If true, log orthogonality loss each epoch
 * @param log_train_loss      If true, log training loss each epoch
 * @param log_test_loss       If true, log test loss each epoch
 * @param log_sparsity        If true, log sparsity of W and H each epoch
 * @param num_threads         Number of OpenMP threads to use
 * @param W                   Basis matrix (n x k), updated in-place
 * @param MaskMatrix          User-supplied mask matrix (n x m), 1s indicate masked entries
 * @return Rcpp::List         List with W, d (scaling), H, log (metrics), and params (run parameters)
 */
template <typename MatrixType, typename MaskMatrixType, bool MaskZeroEntries, bool MaskTestSet, bool MaskMaskMatrix>
Rcpp::List als_nmf_core(
  const MatrixType& V,
  const int k,
  const RandomSparseBinaryMatrix& TestMatrix,
  const size_t epochs,
  const float tol,
  const bool verbose,
  const float L1_W,
  const float L1_H,
  const float L2_W,
  const float L2_H,
  const float ortho_W,
  const float ortho_H,
  const bool log_ortho_loss,
  const bool log_train_loss,
  bool log_test_loss,
  const bool log_sparsity,
  const int num_threads,
  Eigen::MatrixXf& W,
  const MaskMatrixType& MaskMatrix
) {
    // --- Initialization ---
    const int n = V.cols();
    const MatrixType Vt = V.transpose(); // Transpose of V for W update
    MaskMatrixType MaskMatrix_t; // Transposed mask (if needed)
    auto TestMatrix_t = TestMatrix.transpose(); // Transposed test mask

    // Transpose mask if mask matrix is used
    if constexpr(MaskMaskMatrix) {
        MaskMatrix_t = MaskMatrix.transpose();
    }

    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(k, n); // Encoding matrix H
    Eigen::VectorXf d(k); // Scaling vector
    float conv = 1.0f; // Convergence metric

    // If test set is defined and test loss is logged, also log train loss
    if constexpr(MaskTestSet) {
        if (log_train_loss) log_test_loss = true;
    }

    Logger params, logger;

    // Log parameters for reproducibility
    log_params(params, k, epochs, tol, L1_W, L1_H, L2_W, L2_H, ortho_W, ortho_H, TestMatrix, num_threads);

    // --- ALS main loop ---
    for (uint16_t iter = 0; iter < epochs && conv > tol; ++iter) {
        Eigen::MatrixXf W_prev = W;
        Eigen::MatrixXf H_prev = H;

        // --- H update: solve min ||V - W H||^2 ---
        nnls<MatrixType, MaskMatrixType, MaskZeroEntries, MaskTestSet, MaskMaskMatrix>(
            V, W, H, L1_H, L2_H, ortho_H, num_threads, TestMatrix, MaskMatrix
        );
        scale(H, d); // Normalize H and update scaling vector

        // --- W update: solve min ||V^T - H W^T||^2 ---
        nnls<MatrixType, MaskMatrixType, MaskZeroEntries, MaskTestSet, MaskMaskMatrix>(
            Vt, H, W, L1_W, L2_W, ortho_W, num_threads, TestMatrix_t, MaskMatrix_t
        );
        scale(W, d); // Normalize W and update scaling vector

        conv = convergence(W, W_prev); // Check convergence

        // Log values and compute losses
        log_and_compute_losses<MatrixType, MaskMatrixType, MaskZeroEntries, MaskTestSet, MaskMaskMatrix>(
            logger, V, W, H, d, ortho_W, TestMatrix, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity,
            iter, conv, MaskMatrix, num_threads
        );

        if (verbose) verbose_epoch(logger); // Print progress if verbose
        Rcpp::checkUserInterrupt(); // Allow user to interrupt from R
    }

    // --- Sort W, H, d by diagonal before returning ---
    sort_by_diagonal(W, H, d);

    // --- Return results ---
    return Rcpp::List::create(
        Rcpp::Named("W") = W.transpose(),
        Rcpp::Named("d") = d,
        Rcpp::Named("H") = H,
        Rcpp::Named("log") = logger.to_dataframe(),
        Rcpp::Named("params") = params.to_dataframe()
    );
}

#endif // ALS_NMF_H