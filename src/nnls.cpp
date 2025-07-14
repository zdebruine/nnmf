#include "nnmf.h"

inline void nnls(Eigen::MatrixXf& a, Eigen::VectorXf& b, Eigen::MatrixXf& x, const size_t col, const float L1 = 0, const float L2 = 0) {
    float tol = 1;
    for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8; ++it) {
        tol = 0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            float diff = b(i) / a(i, i);
            if (L1 != 0) diff -= L1;
            if (L2 != 0) diff += L2 * x(i, col);
            if (-diff > x(i, col)) {
                if (x(i, col) != 0) {
                    b -= a.col(i) * -x(i, col);
                    tol = 1;
                    x(i, col) = 0;
                }
            } else if (diff != 0) {
                x(i, col) += diff;
                b -= a.col(i) * diff;
                tol += std::abs(diff / (x(i, col) + 1e-15));
            }
        }
    }
}
