// nnmf.h: Common macros and includes for NNMF project
#ifndef RCPPML_H
#define RCPPML_H

// Threshold for switching to sparse optimization
// If the fraction of masked entries is below this, use sparse optimization
#define SPARSE_OPTIMIZATION_THRESHOLD_FOR_MASKING 0.2f

// --- turn off `-Wignored-attributes` just for the noisy code ------------
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif

#ifndef _OMP_H
//[[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

#endif // RCPPML_H