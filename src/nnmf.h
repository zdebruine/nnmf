#ifndef NNMF_H
#define NNMF_H

// --- turn off `-Wignored-attributes` just for the noisy code ------------
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

// headers / code that trigger the warning
//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
// -----------------------------------------------------------------------

#ifndef _OMP_H
//[[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

#endif // NNMF_H