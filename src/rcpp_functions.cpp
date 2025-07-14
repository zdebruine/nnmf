#include "als_nmf.cpp"

// [[Rcpp::export]]
Rcpp::List cpp_als_nnmf_dense(Eigen::MatrixXf V,
  int k,
  uint32_t inv_test_size,
  uint32_t test_seed,
  Eigen::MatrixXf W,
  float tol,
  size_t epochs,
  bool verbose,
  Rcpp::NumericVector L1,
  Rcpp::NumericVector L2,
  Rcpp::NumericVector ortho,
  bool log_train_loss,
  bool log_test_loss,
  int num_threads
) {
    return als_nmf_core(V, k, inv_test_size, test_seed, epochs, tol, verbose, L1, L2, ortho, log_train_loss, log_test_loss, num_threads, W);
}

// [[Rcpp::export]]
Rcpp::List cpp_als_nnmf_sparse(Eigen::SparseMatrix<float> V,
  int k,
  uint32_t inv_test_size,
  uint32_t test_seed,
  Eigen::MatrixXf W,
  float tol,
  size_t epochs,
  bool verbose,
  Rcpp::NumericVector L1,
  Rcpp::NumericVector L2,
  Rcpp::NumericVector ortho,
  bool log_train_loss,
  bool log_test_loss,
  int num_threads
) {
    return als_nmf_core(V, k, inv_test_size, test_seed, epochs, tol, verbose, L1, L2, ortho, log_train_loss, log_test_loss, num_threads, W);
}
