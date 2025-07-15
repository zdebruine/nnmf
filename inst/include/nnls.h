#ifndef NNLS_H
#define NNLS_H

#include "scd_ls.h"
#include "logger.h"
#include "helpers.h"

/**
 * @brief Solve for H in min_{H >= 0} ||V - W H||^2 + reg, using masked cross-validation if specified.
 * 
 * Now returns the mean residual norm after fitting all columns.
 *
 * This function solves the non-negative least squares (NNLS) problem for each column of H:
 *   min_{h_j >= 0} ||V_j - W h_j||^2 + reg
 * where V_j is the j-th column of V, and reg includes L1/L2/orthogonality penalties.
 *
 * If inv_test_size > 0, a random speckled mask is applied to hold out a subset of entries for cross-validation.
 * The mask is generated using a seeded RNG, and the masked entries are excluded from the fit by subtracting their
 * contributions from the right-hand side and Gram matrix.
 *
 * @tparam MatrixType  Eigen dense or sparse matrix type
 * @param V            Data matrix (n x m)
 * @param W            Basis matrix (n x k)
 * @param H            Encoding matrix (k x m), to be solved
 * @param L1           L1 regularization
 * @param L2           L2 regularization
 * @param ortho_lambda Orthogonality penalty
 * @param threads      Number of OpenMP threads
 * @param inv_test_size Inverse test set density (0 = no masking)
 * @param test_seed    RNG seed for masking
 * @param mask_t       If true, mask by row; else, mask by column
 */
template<typename MatrixType>
inline void nnls(const MatrixType& V, const Eigen::MatrixXf& W,
                 Eigen::MatrixXf& H, float L1, float L2,
                 float ortho_lambda, int threads,
                 uint64_t inv_test_size, rng& seed, bool mask_t) {

  // --- Helper lambdas ---
  auto adjust_b = [](Eigen::VectorXf& b, const MatrixType& V, const Eigen::MatrixXf& W, uint64_t j, const std::vector<uint64_t>& idx) {
    if constexpr (is_sparse<MatrixType>::value) {
      for (typename MatrixType::InnerIterator it(V, j); it; ++it) {
        for (auto i : idx) {
          if ((uint64_t)it.row() == i) {
            b -= W.col(i) * it.value();
            break;
          }
        }
      }
    } else {
      for (auto i : idx) {
        b -= W.col(i) * V(i, j);
      }
    }
  };

  auto adjust_G = [](const Eigen::MatrixXf& G, const Eigen::MatrixXf& W, const std::vector<uint64_t>& idx) -> Eigen::MatrixXf {
    if (idx.empty()) return G;
    Eigen::MatrixXf wsub(W.rows(), idx.size());
    for (size_t s = 0; s < idx.size(); ++s) {
      wsub.col(static_cast<Eigen::Index>(s)) = W.col(idx[s]);
    }
    return G - XXt(wsub);
  };

  auto apply_ortho = [](Eigen::MatrixXf& G, float ortho_lambda) {
    if (ortho_lambda != 0.0f) {
      Eigen::VectorXf diag = G.diagonal();
      G += G * ortho_lambda;
      G.diagonal() = diag;
    }
  };

  // --- Main logic ---
  Eigen::MatrixXf G = XXt(W);
  if (inv_test_size == 0) {
    apply_ortho(G, ortho_lambda);
  }

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads)
  #endif
  for (uint64_t j = 0; j < (uint64_t)H.cols(); ++j) {
    Eigen::VectorXf b = W * V.col(j);

    // if a test set is specified, apply masking by adjusting b and G accordingly
    if(inv_test_size != 0){
      auto idx = get_masked_indices(V, seed, j, inv_test_size, mask_t); // now from helpers.cpp
      adjust_b(b, V, W, j, idx);
      Eigen::MatrixXf G_i = adjust_G(G, W, idx);
      apply_ortho(G_i, ortho_lambda);
      scd_ls(G_i, b, H, j, L1, L2);
    } else {
      scd_ls(G, b, H, j, L1, L2);
    }
  }
}

#endif // NNLS_H