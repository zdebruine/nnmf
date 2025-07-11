#include <RcppEigen.h>

// fast symmetric matrix multiplication, X * X.transpose()
inline Eigen::MatrixXf XXt(const Eigen::MatrixXf& X) {
  Eigen::MatrixXf XXt = Eigen::MatrixXf::Zero(X.rows(), X.rows());
  XXt.selfadjointView<Eigen::Lower>().rankUpdate(X);
  XXt.triangularView<Eigen::Upper>() = XXt.transpose();
  XXt.diagonal().array() += 1e-15f;
  return XXt;
}

// scale rows in w (or h) to unit 2-norm and put previous norms in d
inline void scale(Eigen::MatrixXf& w, Eigen::VectorXf& d) {
  d = w.rowwise().norm();
  d.array() += 1e-15f;
  for (int i = 0; i < w.rows(); ++i) {
    w.row(i) /= d(i);
  }
}

// mean squared reconstruction error of A ~ t(W) %*% H, with 2-norm scaling
inline float mse(const Eigen::MatrixXf& A, const Eigen::MatrixXf& W,
                 const Eigen::MatrixXf& H, const Eigen::VectorXf& d) {
  Eigen::MatrixXf W_scaled = W;
  for (int i = 0; i < W.rows(); ++i) {
    W_scaled.row(i) *= d(i);
  }
  Eigen::MatrixXf diff = A - W_scaled.transpose() * H;
  return diff.squaredNorm() / A.size();
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


