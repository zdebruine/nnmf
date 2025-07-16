#ifndef HELPERS_H
#define HELPERS_H

#include "matrix_traits.h"
#include "RcppML.h"
#include "rng.h"
#include "loss.h"

// Scale each row of w (or h) to unit 2-norm and store previous norms in d
inline void scale(Eigen::MatrixXf& w, Eigen::VectorXf& d) {
  d = w.rowwise().norm();
  d.array() += 1e-15f; // avoid division by zero
  for (int i = 0; i < w.rows(); ++i) {
    w.row(i) /= d(i);
  }
}

// Compute convergence metric between w and w_prev
inline float convergence(const Eigen::MatrixXf& w, const Eigen::MatrixXf& w_prev) {
  return (w - w_prev).norm() / (w_prev.norm() + 1e-15f);
}

// Compute orthogonality loss: ||X * X^T - I||_F^2 / rows
inline float orthogonality_loss(const Eigen::MatrixXf& X) {
  Eigen::MatrixXf XXt_mat = XXt(X);
  XXt_mat.diagonal().array() -= 1.0f; // subtract identity
  return XXt_mat.squaredNorm() / X.rows();
}

// Apply orthogonality penalty to Gram matrix G
inline void apply_ortho(Eigen::MatrixXf& G, float ortho_lambda) {
  if (ortho_lambda != 0.0f) {
    Eigen::VectorXf diag = G.diagonal();
    G += G * ortho_lambda;
    G.diagonal() = diag;
  }
}

// Calculate the sparsity (fraction of zero elements) of a matrix
inline float sparsity(const Eigen::MatrixXf& X) {
  int total = X.size();
  int zeros = (X.array() == 0.0f).count();
  return static_cast<float>(zeros) / total;
}

// Select columns from W where mask == value
inline Eigen::MatrixXf select_columns(const Eigen::MatrixXf& W, const Eigen::VectorXf& mask, float value) {
  int count = (mask.array() == value).count();
  Eigen::MatrixXf wsub(W.rows(), count);
  for (int s = 0, pos = 0; s < mask.size(); ++s) {
    if (mask(s) == value) {
      wsub.col(pos) = W.col(s);
      ++pos;
    }
  }
  return wsub;
}

// Calculate the right-hand side for the NNLS problem
template <typename MatrixType>
Eigen::VectorXf calc_rhs(const MatrixType& V, const Eigen::MatrixXf& W, uint64_t col) {
  // Calculate the right-hand side for the NNLS problem
  Eigen::VectorXf rhs = Eigen::VectorXf::Zero(W.rows());
  if constexpr (is_sparse<MatrixType>::value) {
    for (typename MatrixType::InnerIterator it(V, col); it; ++it) {
      rhs += W.col(it.row()) * it.value();
    }
    return rhs;
  } else {
    return W * V.col(col);
  }
}

// Apply masked update to rhs vector for NNLS
template<typename MatrixType, bool Add>
inline void apply_masked_rhs_update(Eigen::VectorXf& rhs, const MatrixType& V, const Eigen::MatrixXf& W, uint64_t col, const Eigen::VectorXf& mask, float value) {
  // For sparse V, only iterate nonzero entries
  if constexpr (is_sparse<MatrixType>::value) {
    for (typename MatrixType::InnerIterator it(V, col); it; ++it) {
      // Only update rhs for entries matching the mask value
      if (mask(it.row()) == value) {
        if constexpr (Add)
          rhs += W.col(it.row()) * it.value(); // add unmasked
        else
          rhs -= W.col(it.row()) * it.value(); // subtract masked
      }
    }
  } else {
    for (int i = 0; i < mask.size(); ++i) {
      // Only update rhs for entries matching the mask value
      if (mask(i) == value) {
        if constexpr (Add)
          rhs += W.col(i) * V(i, col); // add unmasked
        else
          rhs -= W.col(i) * V(i, col); // subtract masked
      }
    }
  }
}

// Sorts W (k x m), H (k x n), and d (k) by the diagonal of W*H (size k)
inline void sort_by_diagonal(Eigen::MatrixXf& W, Eigen::MatrixXf& H, Eigen::VectorXf& d) {
    // get the sort index of d in descending order 
    std::vector<std::pair<float, int>> diag_idx(W.rows());
    for (int i = 0; i < W.rows(); ++i) {
        diag_idx[i] = {d(i), i}; // pair of (value, index)
    }
    std::sort(diag_idx.begin(), diag_idx.end(), [](const auto& a, const auto& b) {
        return a.first > b.first; // descending order
    });

    Eigen::MatrixXf W_sorted = W;
    Eigen::MatrixXf H_sorted = H;
    Eigen::VectorXf d_sorted = d;
    for (int i = 0; i < W.rows(); ++i) {
        int idx = diag_idx[i].second;
        W_sorted.row(i) = W.row(idx);
        H_sorted.row(i) = H.row(idx);
        d_sorted(i) = d(idx);
    }
    W = W_sorted;
    H = H_sorted;
    d = d_sorted;
}

#endif // HELPERS_H


