#include <als_nmf.h>

// Helper function to dispatch to als_nmf_core with correct template parameters
// Handles all combinations of dense/sparse V and mask, and masking options.
template<typename VType, typename MaskType>
Rcpp::List call_als_nmf_core(
  const VType& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose,
  const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho,
  bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads,
  const MaskType& mask, bool MaskZeros
) {
  // Determine if test masking and mask matrix are used
  const bool MaskTest = (inv_test_size > 0);
  const bool MaskMaskMatrix = mask.rows() == V.rows() && mask.cols() == V.cols();

  // Create random test mask matrix for test loss calculation
  RandomSparseBinaryMatrix TestMatrix(test_seed, inv_test_size, V.rows(), V.cols());

  // Lambda to call als_nmf_core with correct template parameters
  auto call = [&](auto MaskZeros, auto MaskTest, auto MaskMaskMatrix) {
    return als_nmf_core<VType, MaskType, MaskZeros, MaskTest, MaskMaskMatrix>(
      V, k, TestMatrix, epochs, tol, verbose,
      L1[0], L1[1], L2[0], L2[1], ortho[0], ortho[1], log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, W, mask
    );
  };

  // Dispatch to correct template instantiation based on masking options
  if(MaskZeros){
    if(MaskTest){
      if(MaskMaskMatrix) return call(std::true_type{}, std::true_type{}, std::true_type{});
      else return call(std::true_type{}, std::true_type{}, std::false_type{});
    } else {
      if(MaskMaskMatrix) return call(std::true_type{}, std::false_type{}, std::true_type{});
      else return call(std::true_type{}, std::false_type{}, std::false_type{});
    }
  } else {
    if(MaskTest){
      if(MaskMaskMatrix) return call(std::false_type{}, std::true_type{}, std::true_type{});
      else return call(std::false_type{}, std::true_type{}, std::false_type{});
    } else {
      if(MaskMaskMatrix) return call(std::false_type{}, std::false_type{}, std::true_type{});
      else return call(std::false_type{}, std::false_type{}, std::false_type{});
    }
  }
}

// Entry point for dense V and dense mask
// [[Rcpp::export]]
Rcpp::List cpp_als_nmf_dense_densemask(
  const Eigen::MatrixXf& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose,
  const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho,
  bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads,
  const Eigen::MatrixXf& mask, bool mask_zeros
) {
  return call_als_nmf_core<Eigen::MatrixXf, Eigen::MatrixXf>(
    V, k, inv_test_size, test_seed, W, tol, epochs, verbose,
    L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss,
    log_sparsity, num_threads, mask, mask_zeros
  );
}

// Entry point for dense V and sparse mask
// [[Rcpp::export]]
Rcpp::List cpp_als_nmf_dense_sparsemask(
  const Eigen::MatrixXf& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose,
  const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho,
  bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads,
  const Eigen::SparseMatrix<float>& mask, bool mask_zeros
) {
  return call_als_nmf_core<Eigen::MatrixXf, Eigen::SparseMatrix<float>>(
    V, k, inv_test_size, test_seed, W, tol, epochs, verbose,
    L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss,
    log_sparsity, num_threads, mask, mask_zeros
  );
}

// Entry point for sparse V and sparse mask
// [[Rcpp::export]]
Rcpp::List cpp_als_nmf_sparse_sparsemask(
  const Eigen::SparseMatrix<float>& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose,
  const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho,
  bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads,
  const Eigen::SparseMatrix<float>& mask, bool mask_zeros
) {
  return call_als_nmf_core<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>>(
    V, k, inv_test_size, test_seed, W, tol, epochs, verbose,
    L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss,
    log_sparsity, num_threads, mask, mask_zeros
  );
}

// Entry point for sparse V and dense mask
// [[Rcpp::export]]
Rcpp::List cpp_als_nmf_sparse_densemask(
  const Eigen::SparseMatrix<float>& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose,
  const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho,
  bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads,
  const Eigen::MatrixXf& mask, bool mask_zeros
) {
  return call_als_nmf_core<Eigen::SparseMatrix<float>, Eigen::MatrixXf>(
    V, k, inv_test_size, test_seed, W, tol, epochs, verbose,
    L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss,
    log_sparsity, num_threads, mask, mask_zeros
  );
}
