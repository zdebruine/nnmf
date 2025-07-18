// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/RcppML.h"
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// cpp_als_nmf_dense_densemask
Rcpp::List cpp_als_nmf_dense_densemask(const Eigen::MatrixXf& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose, const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho, bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads, const Eigen::MatrixXf& mask, bool mask_zeros);
RcppExport SEXP _RcppML_cpp_als_nmf_dense_densemask(SEXP VSEXP, SEXP kSEXP, SEXP inv_test_sizeSEXP, SEXP test_seedSEXP, SEXP WSEXP, SEXP tolSEXP, SEXP epochsSEXP, SEXP verboseSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP orthoSEXP, SEXP log_ortho_lossSEXP, SEXP log_train_lossSEXP, SEXP log_test_lossSEXP, SEXP log_sparsitySEXP, SEXP num_threadsSEXP, SEXP maskSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf& >::type V(VSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type inv_test_size(inv_test_sizeSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type test_seed(test_seedSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXf& >::type W(WSEXP);
    Rcpp::traits::input_parameter< float >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type ortho(orthoSEXP);
    Rcpp::traits::input_parameter< bool >::type log_ortho_loss(log_ortho_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_train_loss(log_train_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_test_loss(log_test_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_sparsity(log_sparsitySEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_als_nmf_dense_densemask(V, k, inv_test_size, test_seed, W, tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}
// cpp_als_nmf_dense_sparsemask
Rcpp::List cpp_als_nmf_dense_sparsemask(const Eigen::MatrixXf& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose, const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho, bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads, const Eigen::SparseMatrix<float>& mask, bool mask_zeros);
RcppExport SEXP _RcppML_cpp_als_nmf_dense_sparsemask(SEXP VSEXP, SEXP kSEXP, SEXP inv_test_sizeSEXP, SEXP test_seedSEXP, SEXP WSEXP, SEXP tolSEXP, SEXP epochsSEXP, SEXP verboseSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP orthoSEXP, SEXP log_ortho_lossSEXP, SEXP log_train_lossSEXP, SEXP log_test_lossSEXP, SEXP log_sparsitySEXP, SEXP num_threadsSEXP, SEXP maskSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf& >::type V(VSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type inv_test_size(inv_test_sizeSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type test_seed(test_seedSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXf& >::type W(WSEXP);
    Rcpp::traits::input_parameter< float >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type ortho(orthoSEXP);
    Rcpp::traits::input_parameter< bool >::type log_ortho_loss(log_ortho_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_train_loss(log_train_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_test_loss(log_test_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_sparsity(log_sparsitySEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<float>& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_als_nmf_dense_sparsemask(V, k, inv_test_size, test_seed, W, tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}
// cpp_als_nmf_sparse_sparsemask
Rcpp::List cpp_als_nmf_sparse_sparsemask(const Eigen::SparseMatrix<float>& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose, const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho, bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads, const Eigen::SparseMatrix<float>& mask, bool mask_zeros);
RcppExport SEXP _RcppML_cpp_als_nmf_sparse_sparsemask(SEXP VSEXP, SEXP kSEXP, SEXP inv_test_sizeSEXP, SEXP test_seedSEXP, SEXP WSEXP, SEXP tolSEXP, SEXP epochsSEXP, SEXP verboseSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP orthoSEXP, SEXP log_ortho_lossSEXP, SEXP log_train_lossSEXP, SEXP log_test_lossSEXP, SEXP log_sparsitySEXP, SEXP num_threadsSEXP, SEXP maskSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<float>& >::type V(VSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type inv_test_size(inv_test_sizeSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type test_seed(test_seedSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXf& >::type W(WSEXP);
    Rcpp::traits::input_parameter< float >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type ortho(orthoSEXP);
    Rcpp::traits::input_parameter< bool >::type log_ortho_loss(log_ortho_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_train_loss(log_train_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_test_loss(log_test_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_sparsity(log_sparsitySEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<float>& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_als_nmf_sparse_sparsemask(V, k, inv_test_size, test_seed, W, tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}
// cpp_als_nmf_sparse_densemask
Rcpp::List cpp_als_nmf_sparse_densemask(const Eigen::SparseMatrix<float>& V, int k, uint64_t inv_test_size, uint64_t test_seed, Eigen::MatrixXf& W, float tol, size_t epochs, bool verbose, const Rcpp::NumericVector& L1, const Rcpp::NumericVector& L2, const Rcpp::NumericVector& ortho, bool log_ortho_loss, bool log_train_loss, bool log_test_loss, bool log_sparsity, int num_threads, const Eigen::MatrixXf& mask, bool mask_zeros);
RcppExport SEXP _RcppML_cpp_als_nmf_sparse_densemask(SEXP VSEXP, SEXP kSEXP, SEXP inv_test_sizeSEXP, SEXP test_seedSEXP, SEXP WSEXP, SEXP tolSEXP, SEXP epochsSEXP, SEXP verboseSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP orthoSEXP, SEXP log_ortho_lossSEXP, SEXP log_train_lossSEXP, SEXP log_test_lossSEXP, SEXP log_sparsitySEXP, SEXP num_threadsSEXP, SEXP maskSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<float>& >::type V(VSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type inv_test_size(inv_test_sizeSEXP);
    Rcpp::traits::input_parameter< uint64_t >::type test_seed(test_seedSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXf& >::type W(WSEXP);
    Rcpp::traits::input_parameter< float >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type ortho(orthoSEXP);
    Rcpp::traits::input_parameter< bool >::type log_ortho_loss(log_ortho_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_train_loss(log_train_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_test_loss(log_test_lossSEXP);
    Rcpp::traits::input_parameter< bool >::type log_sparsity(log_sparsitySEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_als_nmf_sparse_densemask(V, k, inv_test_size, test_seed, W, tol, epochs, verbose, L1, L2, ortho, log_ortho_loss, log_train_loss, log_test_loss, log_sparsity, num_threads, mask, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RcppML_cpp_als_nmf_dense_densemask", (DL_FUNC) &_RcppML_cpp_als_nmf_dense_densemask, 18},
    {"_RcppML_cpp_als_nmf_dense_sparsemask", (DL_FUNC) &_RcppML_cpp_als_nmf_dense_sparsemask, 18},
    {"_RcppML_cpp_als_nmf_sparse_sparsemask", (DL_FUNC) &_RcppML_cpp_als_nmf_sparse_sparsemask, 18},
    {"_RcppML_cpp_als_nmf_sparse_densemask", (DL_FUNC) &_RcppML_cpp_als_nmf_sparse_densemask, 18},
    {NULL, NULL, 0}
};

RcppExport void R_init_RcppML(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
