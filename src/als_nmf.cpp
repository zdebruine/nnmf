#include "nnmf.h"
#include "logger.cpp"
#include "helpers.cpp"
#include "nnls.cpp"

// Templated predict function for both dense and sparse matrices
template<typename MatrixType>
// Solve for X in min ||M x - b||^2 s.t. x >= 0 for each column (NNLS)
// M: design matrix (fixed factor), b: target vector (data column)
inline void predict(const MatrixType& V, const Eigen::MatrixXf& W,
                    Eigen::MatrixXf& H, float L1, float L2,
                    float ortho_lambda, int threads) {
    // Compute Gram matrix: G = W W^T
    Eigen::MatrixXf G = XXt(W);
    if (ortho_lambda != 0.0f) {
        // Add orthogonality penalty: G += ortho_lambda * G
        G += G * ortho_lambda;
    }
    G.diagonal().setOnes(); // control for numerical stability and ortho penalty

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
    for (size_t j = 0; j < H.cols(); ++j) {
        Eigen::VectorXf b(W.rows());
        if constexpr (is_sparse<MatrixType>::value) {
            // Compute b = W * v_j for sparse V using InnerIterator
            b.setZero();
            for (typename MatrixType::InnerIterator it(V, j); it; ++it) {
                b += W.col(it.row()) * it.value();
            }
        } else {
            // Dense case
            b = W * V.col(j);
        }
        // Solve NNLS: min ||G h_j - b||^2 + reg, s.t. h_j >= 0
        nnls(G, b, H, j, L1, L2);
    }
}

// Unified ALS NMF routine for both dense and sparse matrices
template<typename MatrixType>
Rcpp::List als_nmf_core(const MatrixType& V,
  int k,
  uint32_t inv_test_size,
  uint32_t test_seed,
  size_t epochs,
  float tol,
  bool verbose,
  Rcpp::NumericVector L1,
  Rcpp::NumericVector L2,
  Rcpp::NumericVector ortho,
  bool log_train_loss,
  bool log_test_loss,
  int num_threads,
  Eigen::MatrixXf W
) {
    // V: data matrix, W: basis matrix, H: encoding matrix
    const int n = V.cols();
    const MatrixType Vt = V.transpose();
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(k, n);
    Eigen::VectorXf d(k);
    float conv = 1.0f;

    Logger params;
    Logger logger;

    // Log parameters at the beginning
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

    for (uint16_t iter = 0; iter < epochs && conv > tol; ++iter) {
        Eigen::MatrixXf W_prev = W;
        Eigen::MatrixXf H_prev = H;
        // H update: solve min ||V - W H||^2
        predict(V, W, H, (float)L1[1], (float)L2[1], (float)ortho[1], num_threads);
        scale(H, d);
        // W update: solve min ||V^T - H W^T||^2
        predict(Vt, H, W, (float)L1[0], (float)L2[0], (float)ortho[0], num_threads);
        scale(W, d);

        conv = convergence(W, W_prev);

        float mse_val = NA_REAL, ortho_W = NA_REAL, ortho_H = NA_REAL, loss = NA_REAL;
        if (log_train_loss) {
            mse_val = mse(V, W, H, d);
            ortho_W = ortho[0] != 0.f ? orthogonality_loss(W) : 0.f;
            ortho_H = ortho[1] != 0.f ? orthogonality_loss(H) : 0.f;
            loss = mse_val + ortho[0] * ortho_W + ortho[1] * ortho_H;
        }

        // Log values for this epoch
        logger.push("iter", static_cast<float>(iter + 1));
        logger.push("mse", mse_val);
        logger.push("ortho_W", ortho_W);
        logger.push("ortho_H", ortho_H);
        logger.push("loss", loss);
        logger.push("conv", conv);
        logger.push("sparsity_W", sparsity(W));
        logger.push("sparsity_H", sparsity(H));

        if (verbose) {
            Rprintf("iter %d | tol %.6f | mse %.6f | ortho_W %.6f | ortho_H %.6f | loss %.6f\n",
                    iter + 1, conv, mse_val, ortho_W, ortho_H, loss);
        }
        logger.next_epoch();
        Rcpp::checkUserInterrupt();
    }

    return Rcpp::List::create(Rcpp::Named("W") = W.transpose(),
                              Rcpp::Named("d") = d,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("log") = logger.to_dataframe(),
                              Rcpp::Named("params") = params.to_dataframe());
}

