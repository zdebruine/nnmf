#include <RcppEigen.h>
#include <limits>

// [[Rcpp::export]]
Eigen::MatrixXf xxt(Eigen::MatrixXf x) {
  return x * x.transpose();
}

// fast symmetric matrix multiplication, A * A.transpose()
inline Eigen::MatrixXf AAt(const Eigen::MatrixXf& A) {
  Eigen::MatrixXf AAt = Eigen::MatrixXf::Zero(A.rows(), A.rows());
  AAt.selfadjointView<Eigen::Lower>().rankUpdate(A);
  AAt.triangularView<Eigen::Upper>() = AAt.transpose();
  AAt.diagonal().array() += 1e-15f;
  return AAt;
}

// scale rows in w (or h) to unit 2-norm and put previous norms in d
inline void scale(Eigen::MatrixXf& w, Eigen::VectorXf& d) {
  d = w.rowwise().norm();
  d.array() += 1e-15f;
  for (int i = 0; i < w.rows(); ++i) {
    w.row(i) /= d(i);
  }
}

// mean squared reconstruction error of A ~ t(W) %*% H
inline double mse(const Eigen::MatrixXf& A, const Eigen::MatrixXf& W,
                  const Eigen::MatrixXf& H) {
  Eigen::MatrixXf diff = A - W.transpose() * H;
  return static_cast<double>(diff.squaredNorm()) /
         static_cast<double>(A.size());
}

// NNLS SOLVER OF THE FORM ax=b -------------------------------------------------------------
inline void nnls(Eigen::MatrixXf& a, Eigen::VectorXf& b, Eigen::MatrixXf& x,
                 const size_t col, const float L1 = 0.f,
                 const float L2 = 0.f) {
  float tol = 1.f;
  for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8f; ++it) {
    tol = 0.f;
    for (size_t i = 0; i < static_cast<size_t>(x.rows()); ++i) {
      float diff = b(i) / a(i, i);
      if (L1 != 0.f)
        diff -= L1;
      if (L2 != 0.f)
        diff += L2 * x(i, col);
      if (-diff > x(i, col)) {
        if (x(i, col) != 0.f) {
          b -= a.col(i) * -x(i, col);
          tol = 1.f;
          x(i, col) = 0.f;
        }
      } else if (diff != 0.f) {
        x(i, col) += diff;
        b -= a.col(i) * diff;
        tol += std::abs(diff / (x(i, col) + 1e-15f));
      }
    }
  }
}

// update h given A and w (dense matrices only)
inline void predict(const Eigen::MatrixXf& A, const Eigen::MatrixXf& w,
                    Eigen::MatrixXf& h, const float L1, const float L2,
                    const int threads) {
  Eigen::MatrixXf a = AAt(w);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
  for (int i = 0; i < h.cols(); ++i) {
    Eigen::VectorXf b = w * A.col(i);
    nnls(a, b, h, static_cast<size_t>(i), L1, L2);
  }
}

// simple ALS NMF for dense float matrices
// [[Rcpp::export]]
Rcpp::List als_nmf(Eigen::MatrixXf A, const int k, const double tol = 1e-4,
                   const uint16_t maxit = 100, const bool verbose = true,
                   const bool log_train_loss = false) {
  const int m = A.rows();
  const int n = A.cols();
  Eigen::MatrixXf w = Eigen::MatrixXf::Random(k, m).cwiseAbs();
  Eigen::MatrixXf h = Eigen::MatrixXf::Zero(k, n);
  Eigen::VectorXf d(k);

  Rcpp::NumericVector iter_log;
  Rcpp::NumericVector conv_log;
  Rcpp::NumericVector loss_log;

  double conv = 1.0;
  for (uint16_t iter = 0; iter < maxit && conv > tol; ++iter) {
    Eigen::MatrixXf w_prev = w;
    // update h
    predict(A, w, h, 0.f, 0.f, 1);
    scale(h, d);
    // update w
    predict(A.transpose(), h, w, 0.f, 0.f, 1);
    scale(w, d);

    conv = (w - w_prev).norm() / (w_prev.norm() + 1e-15f);
    double loss = std::numeric_limits<double>::quiet_NaN();
    if (log_train_loss) {
      loss = mse(A, w, h);
      loss_log.push_back(loss);
    }
    iter_log.push_back(iter + 1);
    conv_log.push_back(conv);
    if (verbose) {
      if (log_train_loss) {
        Rprintf("iter %d | tol %.6f | mse %.6f\n", iter + 1, conv, loss);
      } else {
        Rprintf("iter %d | tol %.6f\n", iter + 1, conv);
      }
    }
    Rcpp::checkUserInterrupt();
  }

  Rcpp::DataFrame log_df;
  if (log_train_loss)
    log_df = Rcpp::DataFrame::create(Rcpp::Named("iter") = iter_log,
                                     Rcpp::Named("loss") = loss_log,
                                     Rcpp::Named("conv") = conv_log);
  else
    log_df = Rcpp::DataFrame::create(Rcpp::Named("iter") = iter_log,
                                     Rcpp::Named("conv") = conv_log);

  return Rcpp::List::create(Rcpp::Named("w") = w,
                            Rcpp::Named("d") = d,
                            Rcpp::Named("h") = h,
                            Rcpp::Named("log") = log_df);
}
