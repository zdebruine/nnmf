#ifndef HELPERS_H
#define HELPERS_H

#include "matrix_traits.h"
#include "nnmf.h"
#include "rng.h"
#include "loss.h"

// scale rows in w (or h) to unit 2-norm and put previous norms in d
inline void scale(Eigen::MatrixXf& w, Eigen::VectorXf& d) {
  d = w.rowwise().norm();
  d.array() += 1e-15f;
  for (int i = 0; i < w.rows(); ++i) {
    w.row(i) /= d(i);
  }
}

// convergence using only w and w_prev
inline float convergence(const Eigen::MatrixXf& w, const Eigen::MatrixXf& w_prev) {
  return (w - w_prev).norm() / (w_prev.norm() + 1e-15f);
}

// Helper for orthogonality loss: ||X * X^T - I||_F^2 / rows
inline float orthogonality_loss(const Eigen::MatrixXf& X) {
  Eigen::MatrixXf XXt_mat = XXt(X);
  XXt_mat.diagonal().array() -= 1.0f; // subtract identity
  return XXt_mat.squaredNorm() / X.rows();
}

// Calculate the sparsity (fraction of zero elements) of a matrix
inline float sparsity(const Eigen::MatrixXf& X) {
  int total = X.size();
  int zeros = (X.array() == 0.0f).count();
  return static_cast<float>(zeros) / total;
}

template<typename MatrixType>
std::vector<uint64_t> get_masked_indices(const MatrixType& V, rng& seed, uint64_t col, uint64_t inv_test_size, bool mask_t) {
    std::vector<uint64_t> idx;
    idx.reserve(V.rows() / (inv_test_size ? inv_test_size : 1));
    for (uint64_t i = 0; i < (uint64_t)V.rows(); ++i) {
        if (mask_t ? seed.draw(i, col, inv_test_size) : seed.draw(col, i, inv_test_size)) {
            idx.push_back(i);
        }
    }
    return idx;
}

template<typename MatrixType>
void log_and_compute_losses(
    Logger& logger,
    const MatrixType& V,
    const Eigen::MatrixXf& W,
    const Eigen::MatrixXf& H,
    const Eigen::VectorXf& d,
    const Rcpp::NumericVector& ortho,
    uint64_t inv_test_size,
    rng& seed,
    bool log_total_loss,
    bool log_ortho_loss,
    bool log_test_loss,
    bool log_sparsity,
    uint16_t iter,
    float tol
) {
    float total_loss = NA_REAL, ortho_W = NA_REAL, ortho_H = NA_REAL, train_loss = NA_REAL, test_loss = NA_REAL, sparsity_W = NA_REAL, sparsity_H = NA_REAL;
    if (log_ortho_loss) {
        ortho_W = orthogonality_loss(W);
        ortho_H = orthogonality_loss(H);
    }
    if (log_total_loss) {
        total_loss = mse(V, W, d, H);
        if (log_ortho_loss) {
            total_loss += ortho[0] * ortho_W + ortho[1] * ortho_H;
        }
    }
    if (log_test_loss) {
        if (inv_test_size == 0) {
            test_loss = NA_REAL;
            train_loss = total_loss;
        } else {
            test_loss = mse_test(V, W, d, H, seed, inv_test_size);
            if(log_total_loss){
              float prop_test = 1.0f / inv_test_size;
              train_loss = (total_loss - test_loss * prop_test) / (1 - prop_test);
            }
        }
    }
    if (log_sparsity) {
        sparsity_W = sparsity(W);
        sparsity_H = sparsity(H);
    }

    logger.push("iter", static_cast<float>(iter + 1));
    logger.push("total_loss", total_loss);
    logger.push("train_loss", train_loss);
    logger.push("ortho_W", ortho_W);
    logger.push("ortho_H", ortho_H);
    logger.push("test_loss", test_loss);
    logger.push("tol", tol);
    logger.push("sparsity_W", sparsity_W);
    logger.push("sparsity_H", sparsity_H);
    logger.next_epoch();
}

inline void log_params(Logger& params, int k, size_t epochs, float tol,
                       const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2,
                       const Rcpp::NumericVector& ortho, uint64_t inv_test_size,
                       uint64_t test_seed, int num_threads) {
    params.push("k", static_cast<float>(k));
    params.push("epochs", static_cast<float>(epochs));
    params.push("tol", tol);
    params.push("L1_W", (float)L1[0]);
    params.push("L2_W", (float)L2[0]);
    params.push("L1_H", (float)L1[1]);
    params.push("L2_H", (float)L2[1]);
    params.push("ortho_W", (float)ortho[0]);
    params.push("ortho_H", (float)ortho[1]);
    params.push("inv_test_size", static_cast<float>(inv_test_size));
    params.push("test_seed", static_cast<float>(test_seed));
    params.push("num_threads", static_cast<float>(num_threads));
}

inline void verbose_epoch(uint16_t iter, float tol,
                          Logger& logger,
                          uint64_t inv_test_size,
                          bool log_total_loss,
                          bool log_test_loss) {
    Rprintf("iter %d | tol %.6f", iter + 1, tol);
    if (inv_test_size == 0 && log_total_loss) {
        Rprintf(" | total_loss %.6f", logger.last("total_loss"));
    } else if (inv_test_size != 0) {
        if (log_total_loss && !(log_test_loss)) {
            Rprintf(" | total_loss %.6f", logger.last("total_loss"));
        } else {
            if (log_total_loss) Rprintf(" | train_loss %.6f", logger.last("train_loss"));
            if (log_test_loss) Rprintf(" | test_loss %.6f", logger.last("test_loss"));
        }
    }
    Rprintf("\n");
}

#endif // HELPERS_H


