#include <als_nmf.h>

// [[Rcpp::export]]
Rcpp::List cpp_als_nnmf_dense(Eigen::MatrixXf V,
  int k,
  uint64_t inv_test_size,
  uint64_t test_seed,
  Eigen::MatrixXf W,
  float tol,
  size_t epochs,
  bool verbose,
  Rcpp::NumericVector L1,
  Rcpp::NumericVector L2,
  Rcpp::NumericVector ortho,
  bool log_total_loss,
  bool log_ortho_loss,
  bool log_test_loss,
  bool log_sparsity,
  int num_threads
) {
    return als_nmf_core(V, k, inv_test_size, test_seed, epochs, tol, verbose, L1, L2, ortho,
                        log_total_loss, log_ortho_loss, log_test_loss, log_sparsity, num_threads, W);
}

// [[Rcpp::export]]
Rcpp::List cpp_als_nnmf_sparse(Eigen::SparseMatrix<float> V,
  int k,
  uint64_t inv_test_size,
  uint64_t test_seed,
  Eigen::MatrixXf W,
  float tol,
  size_t epochs,
  bool verbose,
  Rcpp::NumericVector L1,
  Rcpp::NumericVector L2,
  Rcpp::NumericVector ortho,
  bool log_total_loss,
  bool log_ortho_loss,
  bool log_test_loss,
  bool log_sparsity,
  int num_threads
) {
    return als_nmf_core(V, k, inv_test_size, test_seed, epochs, tol, verbose, L1, L2, ortho,
                        log_total_loss, log_ortho_loss, log_test_loss, log_sparsity, num_threads, W);
}
