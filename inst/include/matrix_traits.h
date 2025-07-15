#ifndef MATRIX_TRAITS_H
#define MATRIX_TRAITS_H

#include <type_traits>
#include "nnls.h"

// Type trait to detect if a matrix is sparse (has InnerIterator)
template<typename, typename = std::void_t<>>
struct is_sparse : std::false_type {};

template<typename T>
struct is_sparse<T, std::void_t<typename T::InnerIterator>> : std::true_type {};

// fast symmetric matrix multiplication, X * X.transpose()
inline Eigen::MatrixXf XXt(const Eigen::MatrixXf& X) {
  Eigen::MatrixXf XXt = Eigen::MatrixXf::Zero(X.rows(), X.rows());
  XXt.selfadjointView<Eigen::Lower>().rankUpdate(X);
  XXt.triangularView<Eigen::Upper>() = XXt.transpose();
  XXt.diagonal().array() += 1e-15f;
  return XXt;
}

#endif // MATRIX_TRAITS_H
