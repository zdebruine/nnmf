library(testthat)
library(RcppML)

test_that("nmf returns expected output and converges", {
  set.seed(123)
  m <- 10
  n <- 8
  k <- 4
  A <- matrix(runif(m * n), m, n)
  w_init <- matrix(runif(m * k), m, k)
  # Use reasonable defaults for regularization and orthogonality
  res <- nmf(A, k, 0.06125, tol = 1e-4, epochs = 50, verbose = FALSE,
                 L1 = c(0, 0), L2 = c(0, 0), ortho = c(0, 0), log = c("train_loss", "test_loss"),
                 num_threads = 1)
  expect_type(res, "list")
  expect_true(all(dim(res$w) == c(m, k)))
  expect_true(all(dim(res$h) == c(k, n)))
  expect_true(length(res$d) == k)
  expect_true(is.data.frame(res$log))
  # Check convergence: last conv value should be below tolerance
  expect_true(tail(res$log$conv, 1) < 1e-3)
})
