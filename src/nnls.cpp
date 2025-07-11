#include <RcppEigen.h>

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
