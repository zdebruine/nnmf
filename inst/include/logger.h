#ifndef LOGGER_H
#define LOGGER_H

#include "RcppML.h"
#include "rng.h"
#include "helpers.h"

// Logger class for tracking metrics and parameters during ALS NMF
class Logger {
public:
  Logger() : epoch_(0) {}

  // Add a value for a named metric at the current epoch
  void push(const std::string& name, float value) {
    // Add item if not present
    if (logs_.count(name) == 0) {
      logs_[name] = std::vector<float>();
    }
    // Fill previous epochs with NA_REAL if needed
    while ((int)logs_[name].size() < epoch_) {
      logs_[name].push_back(NA_REAL);
    }
    logs_[name].push_back(value);
  }

  // Advance to the next epoch
  void next_epoch() {
    ++epoch_;
  }

  // Get current epoch
  int epoch() const { return epoch_; }

  // Fill missing values for all items for this epoch
  void fill_missing() {
    for (auto& kv : logs_) {
      while ((int)kv.second.size() < epoch_) {
        kv.second.push_back(NA_REAL);
      }
    }
  }

  // Construct Rcpp::DataFrame from logs, always fill missing first
  Rcpp::DataFrame to_dataframe() {
    fill_missing();
    Rcpp::List df;
    for (const auto& kv : logs_) {
      df[kv.first] = kv.second;
    }
    return Rcpp::DataFrame(df);
  }

  // Get the last value for a metric, or NA_REAL if not present
  float last(const std::string& name) const {
    auto it = logs_.find(name);
    if (it == logs_.end() || it->second.empty()) return NA_REAL;
    return it->second.back();
  }

private:
  std::map<std::string, std::vector<float>> logs_;
  int epoch_;
};

// Log and compute losses and other metrics for each epoch
template <typename MatrixType, typename MaskMatrixType, bool MaskZeroEntries, bool MaskTestSet, bool MaskMaskMatrix>
void log_and_compute_losses(
    Logger& logger,
    const MatrixType& V,
    const Eigen::MatrixXf& W,
    const Eigen::MatrixXf& H,
    const Eigen::VectorXf& d,
    const Rcpp::NumericVector& ortho,
    const RandomSparseBinaryMatrix& TestMatrix,
    bool log_ortho_loss,
    bool log_train_loss,
    bool log_test_loss,
    bool log_sparsity,
    uint16_t iter,
    float tol,
    const MaskMatrixType& MaskMatrix,
    const int num_threads
){
    // Initialize all metrics as NA_REAL
    float ortho_W = NA_REAL, ortho_H = NA_REAL, train_loss = NA_REAL, test_loss = NA_REAL, sparsity_W = NA_REAL, sparsity_H = NA_REAL;
    if (log_ortho_loss) {
        ortho_W = orthogonality_loss(W);
        ortho_H = orthogonality_loss(H);
    }
    if (log_train_loss) {
      // test loss will be automatically computed if we are computing train loss and a test set is defined
        std::pair<float, float> both_losses = mse_train<MatrixType, MaskMatrixType, MaskZeroEntries, MaskTestSet, MaskMaskMatrix>(V, W, d, H, TestMatrix, MaskMatrix, num_threads);
        train_loss = both_losses.first;
        if (both_losses.second != NA_REAL) {
            test_loss = both_losses.second;
        }
    } else if (log_test_loss) {
        if (TestMatrix.density() > 0) {
            test_loss = mse_test<MatrixType, MaskMatrixType, MaskZeroEntries, MaskTestSet, MaskMaskMatrix>(V, W, d, H, TestMatrix, MaskMatrix, num_threads);
        }
    }
    if (log_sparsity) {
        sparsity_W = sparsity(W);
        sparsity_H = sparsity(H);
    }

    // Log all metrics for this epoch
    logger.push("iter", static_cast<float>(iter + 1));
    logger.push("train_loss", train_loss);
    logger.push("ortho_W", ortho_W);
    logger.push("ortho_H", ortho_H);
    logger.push("test_loss", test_loss);
    logger.push("tol", tol);
    logger.push("sparsity_W", sparsity_W);
    logger.push("sparsity_H", sparsity_H);
    logger.next_epoch();
}

// Log parameters for reproducibility and diagnostics
inline void log_params(Logger& params, int k, size_t epochs, float tol,
  const float L1_W, const float L1_H,
  const float L2_W, const float L2_H,
  const float ortho_W, const float ortho_H,
  const RandomSparseBinaryMatrix& TestMatrix, int num_threads) {
    params.push("k", static_cast<float>(k));
    params.push("epochs", static_cast<float>(epochs));
    params.push("tol", tol);
    params.push("L1_W", L1_W);
    params.push("L2_W", L2_W);
    params.push("L1_H", L1_H);
    params.push("L2_H", L2_H);
    params.push("ortho_W", ortho_W);
    params.push("ortho_H", ortho_H);
    params.push("test_prop", static_cast<float>(TestMatrix.density()));
    params.push("test_seed", static_cast<float>(TestMatrix.state));
    params.push("num_threads", static_cast<float>(num_threads));
}

// Print progress for the current epoch if verbose is enabled
inline void verbose_epoch(Logger& logger) {
    Rprintf("iter %d | tol %.6f", (int)logger.last("iter"), logger.last("tol"));
    float last_train_loss = logger.last("train_loss");
    float last_test_loss = logger.last("test_loss");
    if(last_train_loss != NA_REAL) {
        Rprintf(" | train_loss %.6f", last_train_loss);
    }
    if(last_test_loss != NA_REAL) {
        Rprintf(" | test_loss %.6f", last_test_loss);
    }
    Rprintf("\n");
}

#endif // LOGGER_H