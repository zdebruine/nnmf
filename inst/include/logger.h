#ifndef LOGGER_H
#define LOGGER_H

#include "nnmf.h"

class Logger {
public:
  Logger() : epoch_(0) {}

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

  void next_epoch() {
    ++epoch_;
  }

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

  float last(const std::string& name) const {
    auto it = logs_.find(name);
    if (it == logs_.end() || it->second.empty()) return NA_REAL;
    return it->second.back();
  }

private:
  std::map<std::string, std::vector<float>> logs_;
  int epoch_;
};

#endif // LOGGER_H