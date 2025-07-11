#include <RcppEigen.h>

// [[Rcpp::export]]
Eigen::MatrixXf xxt(Eigen::MatrixXf x) {
  return x * x.transpose();
}
