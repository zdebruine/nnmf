#include <RcppEigen.h>

inline void nnls(Eigen::MatrixXf& a, Eigen::VectorXf& b, Eigen::MatrixXf& x,
                 const size_t col, const float L1 = 0.f,
                 const float L2 = 0.f) {
  float tol = 1.f;
  float L2_penalty = 1 + L2;
  for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8f; ++it) {
    tol = 0.f;
    for (size_t i = 0; i < static_cast<size_t>(x.rows()); ++i) {
      // note that a(i, i) is always 1 due to 2-norm scaling
      //float diff = b(i);
      float new_x = (b(i) - L1) / L2_penalty;
      float diff = new_x - x(i, col);
      if (-diff > x(i, col)) {
        if (x(i, col) != 0.f) {
          b -= a.col(i) * -x(i, col);
          tol = 1.f;
          x(i, col) = 0.f;
        }
      } else if (diff != 0.f) {
        x(i, col) = new_x;
        b -= a.col(i) * diff;
        tol += std::abs(diff / (new_x + 1e-15f));
      }
    }
  }
}
