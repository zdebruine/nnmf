//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

//[[Rcpp::plugins(openmp)]]
#include <omp.h>

#include "logger.cpp"
#include "helpers.cpp"
#include "nnls.cpp"

// update h given A and w (dense matrices only)
// Add X_free_prev and ortho_lambda arguments for off-diagonal regularisation
inline void predict(const Eigen::MatrixXf& A, const Eigen::MatrixXf& X_fixed,
                    Eigen::MatrixXf& X_free, const float L1, const float L2,
                    const float ortho_lambda, const int threads) {
  Eigen::MatrixXf AtA = XXt(X_fixed);
  // Pure off-diagonal orthogonality regulariser
  if (ortho_lambda != 0.0f) {
    // shrink off-diagonal elements of AtA by multiplying by ortho_lambda
    AtA += AtA * ortho_lambda;
    AtA.diagonal().setOnes();
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
  for (size_t i = 0; i < X_free.cols(); ++i) {
    Eigen::VectorXf b = X_fixed * A.col(i);
    nnls(AtA, b, X_free, i, L1, L2);
  }
}

// simple ALS NMF for dense float matrices
// [[Rcpp::export]]
Rcpp::List cpp_als_nnmf(Eigen::MatrixXf A,
  const int k,
  const uint32_t inv_test_size,
  const uint32_t test_seed,
  Eigen::MatrixXf w,
  const float tol,
  const size_t epochs,
  const bool verbose,
  Rcpp::NumericVector L1,
  Rcpp::NumericVector L2,
  Rcpp::NumericVector ortho,
  const bool log_train_loss,
  const bool log_test_loss,
  const int num_threads
) {

  const int m = A.rows();
  const int n = A.cols();
  Eigen::MatrixXf h = Eigen::MatrixXf::Zero(k, n);
  Eigen::VectorXf d(k);

  Logger logger;

  float conv = 1.0f;
  for (uint16_t iter = 0; iter < epochs && conv > tol; ++iter) {
    logger.next_epoch();

    Eigen::MatrixXf w_prev = w;
    Eigen::MatrixXf h_prev = h;
    // H update with off-diagonal regulariser
    predict(A, w, h, (float)L1[1], (float)L2[1], (float)ortho[1], num_threads);
    scale(h, d);
    // W update with off-diagonal regulariser
    predict(A.transpose(), h, w, (float)L1[0], (float)L2[0], (float)ortho[0], num_threads);
    scale(w, d);

    conv = convergence(w, w_prev);

    float mse_val = NA_REAL, ortho_w = NA_REAL, ortho_h = NA_REAL, loss = NA_REAL;
    if (log_train_loss) {
      mse_val = mse(A, w, h, d);
      ortho_w = ortho[0] != 0.f ? orthogonality_loss(w) : 0.f;
      ortho_h = ortho[1] != 0.f ? orthogonality_loss(h) : 0.f;
      loss = mse_val + ortho[0] * ortho_w + ortho[1] * ortho_h;
    }

    logger.push("iter", static_cast<float>(iter + 1));
    logger.push("mse", mse_val);
    logger.push("ortho_w", ortho_w);
    logger.push("ortho_h", ortho_h);
    logger.push("loss", loss);
    logger.push("conv", conv);

    if (verbose) {
      Rprintf("iter %d | tol %.6f | mse %.6f | ortho_w %.6f | ortho_h %.6f | loss %.6f\n",
              iter + 1, conv, mse_val, ortho_w, ortho_h, loss);
    }
    Rcpp::checkUserInterrupt();
  }

  return Rcpp::List::create(Rcpp::Named("w") = w.transpose(),
                            Rcpp::Named("d") = d,
                            Rcpp::Named("h") = h,
                            Rcpp::Named("log") = logger.to_dataframe());
}