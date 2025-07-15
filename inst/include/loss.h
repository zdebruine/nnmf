#ifndef LOSS_H
#define LOSS_H

#include "nnmf.h"
#include "matrix_traits.h"
#include "rng.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Mean squared error on masked test set, averaged over all columns
template<typename MatType>
inline float mse_test(const MatType& A,
                     const Eigen::MatrixXf& W,
                     const Eigen::VectorXf& d,
                     const Eigen::MatrixXf& H,
                     rng& seed,
                     uint64_t inv_test_density) {
    // W is k x m, but logic expects m x k
    Eigen::MatrixXf W_scaled = W.transpose(); // now m x k
    for (int i = 0; i < W_scaled.cols(); ++i) {
      W_scaled.col(i) *= d(i);
    }
    int m = H.cols();
    Eigen::VectorXf total_loss(m);

    #ifndef NNMF_DISABLE_OPENMP
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    #endif
    for (int col_idx = 0; col_idx < m; ++col_idx) {
      // Use helper for mask indices
      std::vector<uint64_t> masked_idx = get_masked_indices(A, seed, col_idx, inv_test_density, false);

      Eigen::VectorXf pred = W_scaled * H.col(col_idx);
      float s = 0.0f;
      int n = 0;
      if constexpr (is_sparse<MatType>::value) {
        std::unordered_set<uint64_t> mask_set(masked_idx.begin(), masked_idx.end());
        for (int i = 0; i < A.rows(); ++i) {
          if (mask_set.count(i)) {
            float val = 0.0f;
            for (typename MatType::InnerIterator it(A, col_idx); it; ++it) {
              if (it.row() == i) {
                val = it.value();
                break;
              }
            }
            s += std::pow(pred(i) - val, 2);
            ++n;
          }
        }
      } else {
        for (auto i : masked_idx) {
          s += std::pow(pred(i) - A(i, col_idx), 2);
          ++n;
        }
      }
      total_loss(col_idx) = (n > 0 ? s / n : 0.0f);
    }
    return total_loss.array().mean();
}

// Mean squared reconstruction error, averaged over all columns (dense)
template<typename MatType>
inline float mse(const MatType& A,
                     const Eigen::MatrixXf& W,
                     const Eigen::VectorXf& d,
                     const Eigen::MatrixXf& H) {
  // W is k x m, but logic expects m x k
  Eigen::MatrixXf W_scaled = W.transpose(); // now m x k
  for (int i = 0; i < W_scaled.cols(); ++i) {
    W_scaled.col(i) *= d(i);
  }
  int m = H.cols();
  Eigen::VectorXf total_loss(m);

  #ifndef NNMF_DISABLE_OPENMP
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  #endif
  for (int col_idx = 0; col_idx < m; ++col_idx) {
    Eigen::VectorXf pred = W_scaled * H.col(col_idx);
    if constexpr (is_sparse<MatType>::value) {
      for (typename MatType::InnerIterator it(A, col_idx); it; ++it) {
        pred[it.row()] -= it.value();
      }
      total_loss(col_idx) = pred.squaredNorm() / A.rows();
    } else {
      total_loss(col_idx) = (pred.array() - A.col(col_idx).array()).square().mean();
    }
  }
  return total_loss.array().mean();
}

#endif // LOSS_H
