#ifndef SCD_LS_H
#define SCD_LS_H

#include "RcppML.h"


/**
 * @brief Sequential Coordinate Descent for Non-negative Least Squares (NNLS) with L1/L2 regularization.
 *
 * Solves for a single column of X in the following minimization problem:
 *   min_{x >= 0} 0.5 * ||b - A x||^2 + L1 * ||x||_1 + 0.5 * L2 * ||x||^2
 * where:
 *   - A is the Gram matrix (k x k, symmetric positive definite)
 *   - b is the right-hand side vector (k)
 *   - X is the solution matrix (k x m), only column 'col' is updated
 *   - L1, L2 are regularization parameters
 *
 * The algorithm iteratively updates each coordinate of x to minimize the objective,
 * enforcing non-negativity and regularization. After fitting, the solution is stored in X(:, col).
 *
 * @param Gram   Gram matrix (k x k)
 * @param rhs    Right-hand side vector (k)
 * @param X      Solution matrix (k x m), column 'col' is updated in-place
 * @param col    Index of column in X to update
 * @param L1     L1 regularization parameter
 * @param L2     L2 regularization parameter
 */
inline void scd_ls(Eigen::MatrixXf& Gram, Eigen::VectorXf& rhs, Eigen::MatrixXf& X, size_t col, float L1 = 0.0f, float L2 = 0.0f) {
    float tol = 1.0f;
    for (uint8_t it = 0; it < 100 && (tol / rhs.size()) > 1e-8; ++it) {
        tol = 0.0f;
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            float diff = rhs(i) / Gram(i, i);
            if (L1 != 0.0f) diff -= L1;
            if (L2 != 0.0f) diff += L2 * X(i, col);
            if (-diff > X(i, col)) {
                if (X(i, col) != 0.0f) {
                    rhs -= Gram.col(i) * -X(i, col);
                    tol = 1.0f;
                    X(i, col) = 0.0f;
                }
            } else if (diff != 0.0f) {
                X(i, col) += diff;
                rhs -= Gram.col(i) * diff;
                tol += std::abs(diff / (X(i, col) + 1e-15f));
            }
        }
    }
}

#endif
