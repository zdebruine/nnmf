#ifndef ALS_NMF_H
#define ALS_NMF_H

#include "nnls.h"

/**
 * @brief Alternating Least Squares (ALS) Non-negative Matrix Factorization (NMF) core routine.
 *
 * This function performs NMF using ALS, alternating between solving for encoding (H) and basis (W) matrices
 * using non-negative least squares (NNLS). Supports dense and sparse matrices, regularization, orthogonality,
 * and masked cross-validation.
 *
 * @tparam MatrixType         Eigen dense or sparse matrix type
 * @param V                  Data matrix (n x m)
 * @param k                  Number of components (rank)
 * @param inv_test_size      Inverse test set density (0 = no masking)
 * @param test_seed          RNG seed for masking
 * @param epochs             Number of ALS epochs
 * @param tol                Convergence tolerance
 * @param verbose            Print progress to console
 * @param L1                 L1 regularization (W, H)
 * @param L2                 L2 regularization (W, H)
 * @param ortho              Orthogonality penalty (W, H)
 * @param log_total_loss     Log total loss each epoch
 * @param log_ortho_loss     Log orthogonality loss each epoch
 * @param log_train_loss     Log training loss each epoch
 * @param log_test_loss      Log test loss each epoch (not used here)
 * @param num_threads        Number of OpenMP threads
 * @param W                  Initial basis matrix (n x k)
 * @return Rcpp::List        Contains W, d, H, log, params
 */
template<typename MatrixType>
Rcpp::List als_nmf_core(const MatrixType& V,
  int k,
  uint64_t inv_test_size,
  uint64_t test_seed,
  size_t epochs,
  float tol,
  bool verbose,
  Rcpp::NumericVector L1,
  Rcpp::NumericVector L2,
  Rcpp::NumericVector ortho,
  bool log_total_loss,
  bool log_ortho_loss,
  bool log_test_loss,
  bool log_sparsity,
  int num_threads,
  Eigen::MatrixXf W
) {
    // --- Initialization ---
    const int n = V.cols();
    const MatrixType Vt = V.transpose();
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(k, n); // encoding matrix
    Eigen::VectorXf d(k); // scaling vector
    float conv = 1.0f; // convergence metric

    Logger params; // parameter logger
    Logger logger; // epoch logger
    rng test_rng(test_seed); // random number generator for masking

    // Use helper function for logging parameters
    log_params(params, k, epochs, tol, L1, L2, ortho, inv_test_size, test_seed, num_threads);

    // --- ALS main loop ---
    for (uint16_t iter = 0; iter < epochs && conv > tol; ++iter) {
        Eigen::MatrixXf W_prev = W;
        Eigen::MatrixXf H_prev = H;
        // --- H update: solve min ||V - W H||^2 ---
        nnls(V, W, H, (float)L1[1], (float)L2[1], (float)ortho[1], num_threads, inv_test_size, test_rng, false);
        scale(H, d); // normalize H and update scaling vector

        // --- W update: solve min ||V^T - H W^T||^2 ---
        nnls(Vt, H, W, (float)L1[0], (float)L2[0], (float)ortho[0], num_threads, inv_test_size, test_rng, true);
        scale(W, d); // normalize W and update scaling vector

        conv = convergence(W, W_prev); // check convergence

        // Log values
        log_and_compute_losses(
            logger, V, W, H, d, ortho, inv_test_size, test_rng,
            log_total_loss, log_ortho_loss, log_test_loss, log_sparsity,
            iter, tol
        );

        // --- Verbose output ---
        if (verbose) {
            verbose_epoch(iter, conv, logger, inv_test_size, log_total_loss, log_test_loss);
        }

        Rcpp::checkUserInterrupt(); // allow user to interrupt
    }

    // --- Return results ---
    return Rcpp::List::create(Rcpp::Named("W") = W.transpose(),
                              Rcpp::Named("d") = d,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("log") = logger.to_dataframe(),
                              Rcpp::Named("params") = params.to_dataframe());
}

#endif // ALS_NMF_H