#ifndef SCD_LS_H
#define SCD_LS_H

#include "nnmf.h"

/**
 * @brief Sequential Coordinate Descent for Non-negative Least Squares with L1/L2 regularization.
 *
 * This routine solves for x in the following minimization problem for a single column:
 *     min_x 0.5 * ||b - A x||^2 + L1 * ||x||_1 + 0.5 * L2 * ||x||^2
 * subject to x >= 0
 * where:
 *   - A is a symmetric positive definite matrix (Gram matrix)
 *   - b is the right-hand side vector (modified in-place to hold residuals)
 *   - x is the solution vector (non-negative)
 *   - L1, L2 are regularization parameters
 *
 * The algorithm iteratively updates each coordinate of x to minimize the objective,
 * enforcing non-negativity and regularization. After fitting, it returns the norm
 * of the residuals (||b||), since b is updated in-place to b - A x.
 *
 * @param a    Gram matrix (square, symmetric, positive definite)
 * @param b    Right-hand side vector (modified in-place)
 * @param x    Solution matrix (updated in-place)
 * @param col  Column of x to update
 * @param L1   L1 regularization parameter
 * @param L2   L2 regularization parameter
 */
inline void scd_ls(Eigen::MatrixXf& a, Eigen::VectorXf& b, Eigen::MatrixXf& x, const size_t col, const float L1 = 0, const float L2 = 0) {
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

#endif
