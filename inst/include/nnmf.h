// nnmf.h: Common macros and includes for NNMF project
#ifndef NNMF_H
#define NNMF_H

// Macro to disable OpenMP (define NNMF_DISABLE_OPENMP to disable)
#ifdef NNMF_DISABLE_OPENMP
#undef _OPENMP
#endif

// Macro to enable debug bounds checking (define NNMF_DEBUG_BOUNDS to enable)
#ifdef NNMF_DEBUG_BOUNDS
#include <stdexcept>
#include <cassert>
#define NNMF_BOUNDS_ASSERT(expr, msg) \
    do { \
        if (!(expr)) { \
            throw std::out_of_range(msg); \
        } \
    } while (0)
#else
#define NNMF_BOUNDS_ASSERT(expr, msg) ((void)0)
#endif

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