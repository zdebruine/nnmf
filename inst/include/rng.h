#ifndef RNG_H
#define RNG_H

// xorshift64 Linear Congruential Generator for reproducible random numbers
#pragma once
#include <cstdint>
#include <cmath>

// Simple RNG class for reproducible pseudo-random numbers
class rng {
   private:
    uint64_t state;

   public:
    rng(uint64_t state) : state(state) {}

    // Advance the RNG state
    void advance_state() {
        state ^= state << 19;
        state ^= state >> 7;
        state ^= state << 36;
    }

    // Get current state
    uint64_t operator*() const {
        return state;
    }

    // Generate a random number based on internal state
    uint64_t rand() {
        uint64_t x = state ^ (state << 38);
        x ^= x >> 13;
        x ^= x << 23;
        return x;
    }

    // Generate a random number based on input i
    uint64_t rand(uint64_t i) {
        // advance i
        i ^= i << 19;
        i ^= i >> 7;
        i ^= i << 36;

        // add i to state
        uint64_t x = state + i;

        // advance state
        x ^= x << 38;
        x ^= x >> 13;
        x ^= x << 23;

        return x;
    }

    // Generate a random number based on inputs i and j
    uint64_t rand(uint64_t i, uint64_t j) {
        uint64_t x = rand(i);

        // advance j
        j ^= j >> 7;
        j ^= j << 23;
        j ^= j >> 8;

        // add j to state
        x += j;

        // advance state
        x ^= x >> 7;
        x ^= x << 53;
        x ^= x >> 4;

        return x;
    }

    // Sample an integer in [0, max_value)
    template <typename T>
    T sample(T max_value) {
        return rand() % max_value;
    }

    // Sample an integer in [0, max_value) using i
    template <typename T>
    T sample(uint64_t i, T max_value) {
        return rand(i) % max_value;
    }

    // Sample an integer in [0, max_value) using i and j
    template <typename T>
    T sample(uint64_t i, uint64_t j, T max_value) {
        return rand(i, j) % max_value;
    }

    // Draw a boolean with probability 1/probability
    template <typename T>
    bool draw(T probability) {
        return sample(probability) == 0;
    }

    template <typename T>
    bool draw(uint64_t i, T probability) {
        return sample(i, probability) == 0;
    }

    template <typename T>
    bool draw(uint64_t i, uint64_t j, T probability) {
        return sample(i, j, probability) == 0;
    }

    // Uniform float in [0,1)
    template <typename T>
    float uniform() {
        T x = (T)rand() / UINT64_MAX;
        return x - std::floor(x);
    }

    template <typename T>
    float uniform(uint64_t i) {
        T x = (T)rand(i) / UINT64_MAX;
        return x - std::floor(x);
    }

    template <typename T>
    float uniform(uint64_t i, uint64_t j) {
        T x = (T)rand(i, j) / UINT64_MAX;
        return x - std::floor(x);
    }
};

// RandomSparseBinaryMatrix: efficiently generates a reproducible random binary mask
class RandomSparseBinaryMatrix {
   public:
    uint64_t state;
    uint8_t inv_test_size;
    size_t rows_, cols_;
    bool is_transposed;

    // Constructor
    RandomSparseBinaryMatrix(uint64_t state, uint8_t inv_test_size, size_t rows, size_t cols, bool is_transposed = false)
        : state(state), inv_test_size(inv_test_size), rows_(rows), cols_(cols), is_transposed(is_transposed) {}

    // Returns true if (i,j) is in the random mask
    bool operator()(uint64_t i, uint64_t j) const {
        // Advance i and j for randomness
        i ^= i << 19;
        i ^= i >> 7;
        i ^= i << 36;

        j ^= j >> 7;
        j ^= j << 23;
        j ^= j >> 8;

        uint64_t x = state;
        if (is_transposed) {
            // add i to state
            x += i;

            // advance state
            x ^= x << 38;
            x ^= x >> 13;
            x ^= x << 23;

            // add j to state
            x += j;
        } else {
            // add j to state
            x += j;

            // advance state
            x ^= x << 38;
            x ^= x >> 13;
            x ^= x << 23;

            // add i to state
            x += i;
        }

        // advance state
        x ^= x >> 7;
        x ^= x << 53;
        x ^= x >> 4;

        return (x % inv_test_size) == 0;
    }

    // Number of rows
    size_t rows() const { return rows_; }
    // Number of columns
    size_t cols() const { return cols_; }

    // Get a column as a dense Eigen vector
    Eigen::VectorXf col(uint64_t col) const {
        Eigen::VectorXf vec(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            vec(i) = operator()(i, col) ? 1.0f : 0.0f;
        }
        return vec;
    }

    // InnerIterator for iterating over nonzero entries in a column
    class InnerIterator {
        public:
            InnerIterator(RandomSparseBinaryMatrix& ptr, size_t col)
                : ptr(ptr), col_(col), row_(0) {
                // Advance to first nonzero
                while(row_ < ptr.rows() && !ptr.operator()(row_, col_)) ++row_;
            }
            operator bool() const { return (row_ < ptr.rows()); }
            InnerIterator& operator++() {
                ++row_;
                while(row_ < ptr.rows() && !ptr.operator()(row_, col_)) ++row_;
                return *this;
            }
            double value() const { return ptr.operator()(row_, col_) && row_ < ptr.rows(); }
            size_t row() const { return row_; }
            size_t col() const { return col_; }

        private:
            RandomSparseBinaryMatrix& ptr;
            size_t col_, row_;
    };

    // Return a transposed view of the mask
    RandomSparseBinaryMatrix transpose() const {
        return RandomSparseBinaryMatrix(state, inv_test_size, cols_, rows_, !is_transposed);
    }

    // Return the density of the mask
    float density() const {
        return 1.0f / inv_test_size;
    }
};

#endif // RNG_H