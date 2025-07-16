#ifndef NNLS_H
#define NNLS_H

#include "scd_ls.h"
#include "logger.h"
#include "helpers.h"

/**
 * @brief Solve the Non-negative Least Squares (NNLS) problem for each column of H, with optional masking and regularization.
 *
 * This function solves for H in the following problem:
 *   min_{H >= 0} ||V - W H||^2 + L1 * ||H||_1 + L2 * ||H||_2^2 + ortho * orthogonality_penalty(H)
 *
 * Supports masking of zero entries, random test sets, and user-supplied mask matrices. Handles both dense and sparse matrices.
 *
 * @tparam MatrixType         Eigen dense or sparse matrix type for V
 * @tparam MaskMatrixType     Eigen dense or sparse matrix type for mask
 * @tparam MaskZeroEntries    If true, mask out zero entries in V
 * @tparam MaskTestSet        If true, apply a random test set mask for cross-validation
 * @tparam MaskMaskMatrix     If true, apply a user-supplied mask matrix
 * @param V                   Data matrix (n x m)
 * @param W                   Basis matrix (n x k)
 * @param H                   Encoding matrix (k x m), to be solved (output)
 * @param L1                  L1 regularization parameter for H
 * @param L2                  L2 regularization parameter for H
 * @param ortho               Orthogonality penalty for H
 * @param num_threads         Number of OpenMP threads to use
 * @param TestMatrix          RandomSparseBinaryMatrix for test set masking
 * @param MaskMatrix          User-supplied mask matrix (n x m), 1s indicate masked entries
 *
 * This function dispatches to different masking regimes and calls scd_ls() for each column.
 */
template <typename MatrixType, typename MaskMatrixType, bool MaskZeroEntries, bool MaskTestSet, bool MaskMaskMatrix>
inline void nnls(const MatrixType& V, const Eigen::MatrixXf& W,
                 Eigen::MatrixXf& H, float L1, float L2,
                 float ortho, int num_threads,
                 const RandomSparseBinaryMatrix& TestMatrix, const MaskMatrixType& MaskMatrix) {

  constexpr int mask_count = (MaskZeroEntries ? 1 : 0) + (MaskTestSet ? 1 : 0) + (MaskMaskMatrix ? 1 : 0);

  // Precompute Gram matrix for W
  Eigen::MatrixXf Gram = XXt(W);
  if constexpr (mask_count == 0) apply_ortho(Gram, ortho);

  // Main loop: solve for each column of H
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads)
  #endif
  for (uint64_t col = 0; col < static_cast<uint64_t>(H.cols()); ++col) {
    Eigen::VectorXf rhs;
    if constexpr (mask_count == 0) {
      // No masking: standard NNLS
      rhs = W * V.col(col);
      scd_ls(Gram, rhs, H, col, L1, L2);
      continue;
    }

    // Compute mask vector for this column (1 = masked, 0 = unmasked)
    Eigen::MatrixXf Gram_col;
    Eigen::VectorXf mask_vec = compute_mask_vector<MatrixType, MaskMatrixType, MaskZeroEntries, MaskTestSet, MaskMaskMatrix>(V, col, TestMatrix, MaskMatrix);
    float frac_masked = (mask_vec.array() > 0).count() / static_cast<float>(mask_vec.size());

    // Branch on masking regime
    if (frac_masked == 0.0f) {
      // --- No masking for this column ---
      rhs = calc_rhs(V, W, col);
      Gram_col = Gram;
    } else if (frac_masked < SPARSE_OPTIMIZATION_THRESHOLD_FOR_MASKING) {
      // --- Light masking: exclude a small fraction of entries ---
      rhs = calc_rhs(V, W, col);
      apply_masked_rhs_update<MatrixType, false>(rhs, V, W, col, mask_vec, 1.0f); // mask_vec == 1: subtract
      Eigen::MatrixXf masked_cols = select_columns(W, mask_vec, 1.0f);
      Gram_col = Gram - XXt(masked_cols);
    } else {
      // --- Heavy masking: only use unmasked entries ---
      rhs = Eigen::VectorXf::Zero(W.rows());
      apply_masked_rhs_update<MatrixType, true>(rhs, V, W, col, mask_vec, 0.0f); // mask_vec == 0: add
      Eigen::MatrixXf unmasked_cols = select_columns(W, mask_vec, 0.0f);
      Gram_col = XXt(unmasked_cols);
    }
    apply_ortho(Gram_col, ortho);
    scd_ls(Gram_col, rhs, H, col, L1, L2);
  }
}

#endif // NNLS_H