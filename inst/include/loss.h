#ifndef LOSS_H
#define LOSS_H

#include "RcppML.h"
#include "matrix_traits.h"
#include "rng.h"
#include "helpers.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Compute a dense mask vector for a given column, using template booleans for masking types
template <typename MatrixType, typename MaskMatrixType, bool MaskZeroEntries, bool MaskTestSet, bool MaskMaskMatrix>
Eigen::VectorXf compute_mask_vector(const MatrixType& V, uint64_t col, const RandomSparseBinaryMatrix& TestMatrix, const MaskMatrixType& mask) {
  Eigen::VectorXf mask_vec = Eigen::VectorXf::Zero(V.rows());

  // 1. Mask zero entries in V
  if constexpr (MaskZeroEntries) {
    if constexpr (is_sparse<MatrixType>::value) {
      mask_vec.setOnes();
      for (typename MatrixType::InnerIterator it(V, col); it; ++it) {
        mask_vec(it.row()) = 0.0f;
      }
    } else {
      mask_vec = (V.col(col).array() == 0.0f).template cast<float>();
    }
  }

  // 2. Mask random test set
  if constexpr (MaskTestSet) {
    mask_vec += TestMatrix.col(col);
  }

  // 3. Mask user-supplied mask matrix
  if constexpr (MaskMaskMatrix) {
    if constexpr (is_sparse<MaskMatrixType>::value) {
      for (typename MaskMatrixType::InnerIterator it(mask, col); it; ++it) {
        mask_vec(it.row()) = 1.0f;
      }
    } else {
      mask_vec += mask.col(col);
    }
  }

  // Ensure mask vector is binary (0 or 1)
  for(auto& val : mask_vec){
    if(val != 0.0f) val = 1.0f; // convert any non-zero to 1
  }
  return mask_vec;
}

/**
 * @brief Compute mean squared error (MSE) on a masked test set for NMF models.
 *
 * Calculates the average MSE over all columns, using only entries selected by the test mask and excluded by other masking logic.
 * Supports dense/sparse matrices, masking of zero entries, random test set masking, and user-supplied mask matrices.
 *
 * @tparam MatrixType         Eigen dense or sparse matrix type for the data matrix
 * @tparam MatrixType2        Eigen dense or sparse matrix type for the mask
 * @tparam MaskZeroEntries    If true, mask out zero entries in the data matrix
 * @tparam MaskTestSet        If true, apply a random test set mask for cross-validation
 * @tparam MaskMaskMatrix     If true, apply a user-supplied mask matrix
 * @param A                   Data matrix (n x m)
 * @param W                   Basis matrix (n x k)
 * @param d                   Scaling vector (k)
 * @param H                   Encoding matrix (k x m)
 * @param TestMatrix          RandomSparseBinaryMatrix for test set masking
 * @param mask                User-supplied mask matrix (n x m), 1s indicate masked entries
 * @param num_threads         Number of OpenMP threads to use
 * @return float              Mean squared error on the masked test set (averaged over all columns)
 */
template <typename MatrixType, typename MatrixType2, bool MaskZeroEntries, bool MaskTestSet, bool MaskMaskMatrix>
float mse_test(const MatrixType& A,
               const Eigen::MatrixXf& W,
               const Eigen::VectorXf& d,
               const Eigen::MatrixXf& H,
               const RandomSparseBinaryMatrix& TestMatrix,
               const MatrixType2& mask,
               const int num_threads) {
    // Return zero if no test mask is used
    if (TestMatrix.density() == 0) {
        return 0.0f;
    }

    // Scale W by d for reconstruction
    Eigen::MatrixXf W_scaled = W.transpose() * d.asDiagonal();
    Eigen::VectorXf total_loss(H.cols());

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (int col_idx = 0; col_idx < H.cols(); ++col_idx) {
      float s = 0.0f;
      int n = 0;
      // Use template parameters for masking logic
      if constexpr (is_sparse<MatrixType>::value) {
        if constexpr (is_sparse<MatrixType2>::value) {
          typename MatrixType2::InnerIterator mask_it(mask, col_idx);
          typename MatrixType::InnerIterator it(A, col_idx);
          for (int i = 0; i < A.rows(); ++i) {
            if constexpr (MaskTestSet) { if (!TestMatrix(i, col_idx)) continue;}
            while(it && it.row() < i) ++it;
            if constexpr (MaskZeroEntries) { if (!it || it.row() != i) continue; }
            if constexpr (MaskMaskMatrix) { while(mask_it && mask_it.row() < i) ++mask_it; if(mask_it && mask_it.row() == i) continue; }
            if(it && it.row() == i){
              s += std::pow(W_scaled.row(i) * H.col(col_idx) - it.value(), 2);
            } else {
              s += std::pow(W_scaled.row(i) * H.col(col_idx), 2);
            }
            ++n;
          }
        } else {
          // Input is sparse, mask is dense
          typename MatrixType::InnerIterator it(A, col_idx);
          for (int i = 0; i < A.rows(); ++i) {
            if constexpr (MaskTestSet) { if (!TestMatrix(i, col_idx)) continue; }
            while(it && it.row() < i) ++it;
            if constexpr (MaskZeroEntries) { if (!it || it.row() != i) continue; }
            if constexpr (MaskMaskMatrix) { if (mask(i, col_idx) == 1.0f) continue; }
            if(it && it.row() == i){
              s += std::pow(W_scaled.row(i) * H.col(col_idx) - it.value(), 2);
            } else {
              s += std::pow(W_scaled.row(i) * H.col(col_idx), 2);
            }
            ++n;
          }
        }
      } else {
        // input matrix is dense
        if constexpr (is_sparse<MatrixType2>::value) {
          typename MatrixType2::InnerIterator mask_it(mask, col_idx);
          for (int i = 0; i < A.rows(); ++i) {
            if constexpr (MaskTestSet) { if (!TestMatrix(i, col_idx)) continue; }
            if constexpr (MaskZeroEntries) { if (A(i, col_idx) == 0.0f) continue; }
            if constexpr (MaskMaskMatrix) { while(mask_it && mask_it.row() < i) ++mask_it; if(mask_it && mask_it.row() == i) continue; }
            s += std::pow(W_scaled.row(i) * H.col(col_idx) - A(i, col_idx), 2);
            ++n;
          }
        } else {
          // input matrix and mask matrix are dense
          for (int i = 0; i < A.rows(); ++i) {
            if constexpr (MaskTestSet) { if (!TestMatrix(i, col_idx)) continue; }
            if constexpr (MaskMaskMatrix) { if (mask(i, col_idx) == 1.0f) continue; }
            if constexpr (MaskZeroEntries) { if (A(i, col_idx) == 0.0f) continue; }
            s += std::pow(W_scaled.row(i) * H.col(col_idx) - A(i, col_idx), 2);
            ++n;
          }
        }
      }
      total_loss(col_idx) = (n > 0 ? s / n : 0.0f);
    }
    return total_loss.array().mean();
}

/**
 * @brief Compute mean squared error (MSE) for training data, optionally also returning test set MSE.
 *
 * Calculates the MSE over unmasked values in the data matrix, supporting masking of zero entries, test set masking, and user-supplied mask matrices.
 * If a test set is defined, also computes the MSE on the test set entries.
 *
 * @tparam MatrixType         Eigen dense or sparse matrix type for the data matrix
 * @tparam MatrixType2        Eigen dense or sparse matrix type for the mask
 * @tparam MaskZeroEntries    If true, mask out zero entries in the data matrix
 * @tparam MaskTestSet        If true, apply a random test set mask for cross-validation
 * @tparam MaskMaskMatrix     If true, apply a user-supplied mask matrix
 * @param V                   Data matrix (n x m)
 * @param W                   Basis matrix (n x k)
 * @param d                   Scaling vector (k)
 * @param H                   Encoding matrix (k x m)
 * @param TestMatrix          RandomSparseBinaryMatrix for test set masking
 * @param mask                User-supplied mask matrix (n x m), 1s indicate masked entries
 * @param num_threads         Number of OpenMP threads to use
 * @return std::pair<float, float>  Pair of (train MSE, test MSE). If no test set, test MSE is 0.0f.
 */
template <typename MatrixType, typename MatrixType2, bool MaskZeroEntries, bool MaskTestSet, bool MaskMaskMatrix>
std::pair<float, float> mse_train(const MatrixType& V,
                const Eigen::MatrixXf& W,
                const Eigen::VectorXf& d,
                const Eigen::MatrixXf& H,
                const RandomSparseBinaryMatrix& TestMatrix,
                const MatrixType2& mask,
                const int num_threads) {
  // Scale W by d for reconstruction
  Eigen::MatrixXf W_scaled = W.transpose() * d.asDiagonal(); 
  Eigen::VectorXf train_loss(H.cols());
  Eigen::VectorXf test_loss(H.cols());
  Eigen::VectorXi num_in_train(H.cols());
  Eigen::VectorXi num_in_test(H.cols());
  num_in_train.setConstant(H.cols());

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads)
  #endif
  for (int col_idx = 0; col_idx < H.cols(); ++col_idx) {
    // Compute reconstruction error for this column
    Eigen::VectorXf losses = W_scaled * H.col(col_idx);
    if constexpr (is_sparse<MatrixType>::value) {
      for (typename MatrixType::InnerIterator it(V, col_idx); it; ++it) {
        losses(it.row()) -= it.value();
      }
    } else {
      losses.array() -= V.col(col_idx).array();
    }

    // Apply masking logic
    if constexpr (MaskZeroEntries || MaskTestSet || MaskMaskMatrix) {
      if constexpr (MaskTestSet){
        // Calculate test loss simultaneously as this incurs very little overhead
        Eigen::VectorXf unmasked_not_test;
        if constexpr(MaskZeroEntries || MaskMaskMatrix){
          unmasked_not_test = Eigen::VectorXf::Ones(V.rows()) - compute_mask_vector<MatrixType, MatrixType2, MaskZeroEntries, false, MaskMaskMatrix>(V, col_idx, TestMatrix, mask);
        }
        for (int i = 0; i < V.rows(); ++i) {
          if constexpr(MaskZeroEntries || MaskMaskMatrix){
            if (unmasked_not_test(i) == 0) continue;
          }
          if (TestMatrix(i, col_idx)){
            test_loss(col_idx) += losses(i) * losses(i);
            losses(i) = 0.0f;
            num_in_test(col_idx) += 1;
          }
        }
        losses.array() *= unmasked_not_test.array();
        num_in_train(col_idx) -= (num_in_test(col_idx) + (unmasked_not_test.array() == 0).count());
      } else {
        Eigen::VectorXf unmasked_vec = Eigen::VectorXf::Ones(V.rows()) - compute_mask_vector<MatrixType, MatrixType2, MaskZeroEntries, false, MaskMaskMatrix>(V, col_idx, TestMatrix, mask);
        num_in_train(col_idx) = (unmasked_vec.array() > 0).count();
        losses.array() *= unmasked_vec.array();
      }
    }
    train_loss(col_idx) = losses.squaredNorm();
  }
  // Compute total train loss
  float tot_train_loss = train_loss.array().sum() / num_in_train.array().sum();
  auto tot_num_in_test = num_in_test.array().sum();
  
  if constexpr(!MaskTestSet){
    return std::pair<float, float>(tot_train_loss, 0.0f); // no test loss if no entries in test set
  }

  if(tot_num_in_test == 0) {
    return std::pair<float, float>(tot_train_loss, 0.0f); // no test loss if no entries in test set
  } else {
    float tot_test_loss = test_loss.array().sum() / num_in_test.array().sum();
    return std::pair<float, float>(tot_train_loss, tot_test_loss);
  }  
}

#endif // LOSS_H
