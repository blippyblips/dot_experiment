#define PICOBENCH_IMPLEMENT_WITH_MAIN
// #define PICOBENCH_IMPLEMENT
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <new>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include "aligned_allocator.h"

// #include "cl_dot.h" // part 2
#include "picobench.h"

void matrix_multiply_naive(const float* A, const float* B, float* C, size_t n,
                           size_t m, size_t p) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < m; ++k) {
        sum += A[i * m + k] * B[k * p + j];
      }
      C[i * p + j] = sum;
    }
  }
}

void matrix_multiply_unrolled(const float* A, const float* B, float* C,
                              size_t n, size_t m, size_t p) {
  constexpr size_t unroll_factor = 4;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      float sum[unroll_factor] = {0.0f, 0.0f, 0.0f, 0.0f};
      size_t k;
      for (k = 0; k + unroll_factor - 1 < m; k += unroll_factor) {
        for (size_t u = 0; u < unroll_factor; ++u) {
          sum[u] += A[i * m + k + u] * B[(k + u) * p + j];
        }
      }
      float total_sum = sum[0] + sum[1] + sum[2] + sum[3];
      for (; k < m; ++k) {
        total_sum += A[i * m + k] * B[k * p + j];
      }
      C[i * p + j] = total_sum;
    }
  }
}

void matrix_multiply_blocked_v2(std::span<const float> A,
                                std::span<const float> B, std::span<float> C,
                                size_t n, size_t m, size_t p) {
  constexpr size_t BLOCK_SIZE{4};
  for (size_t i = 0; i < n; i += BLOCK_SIZE) {
    for (size_t j = 0; j < p; j += BLOCK_SIZE) {
      for (size_t k = 0; k < m; k += BLOCK_SIZE) {
        for (size_t ii = i; ii < std::min<size_t>(i + BLOCK_SIZE, n); ++ii) {
          for (size_t jj = j; jj < std::min<size_t>(j + BLOCK_SIZE, p); ++jj) {
            float sum = 0;
            for (size_t kk = k; kk < std::min<size_t>(k + BLOCK_SIZE, m);
                 ++kk) {
              sum += A[ii * m + kk] * B[kk * p + jj];
            }
            C[ii * p + jj] += sum;
          }
        }
      }
    }
  }
}

void matrix_multiply_simd(const float* lhs, const float* rhs,
                          float* result, size_t rows_of_a, size_t shared_dimension, size_t cols_of_b) {
  // Assert that input matrices have the expected sizes
  assert(lhs.size() == rows_of_a * shared_dimension);
  assert(rhs.size() == shared_dimension * cols_of_b);
  assert(result.size() == rows_of_a * cols_of_b);

  for (size_t row = 0; row < rows_of_a; ++row) {
    for (size_t col = 0; col < cols_of_b; ++col) {
      __m256 simd_sum = _mm256_setzero_ps();
      size_t k;
      for (k = 0; k + 7 < shared_dimension; k += 8) {
        __m256 simd_a = _mm256_loadu_ps(&lhs[row * shared_dimension + k]);
        __m256 simd_b = _mm256_loadu_ps(&rhs[k * cols_of_b + col]);
        simd_sum = _mm256_fmadd_ps(simd_a, simd_b, simd_sum);
      }
      float total_sum[8];
      _mm256_storeu_ps(total_sum, simd_sum);
      float final_sum = total_sum[0] + total_sum[1] + total_sum[2] +
                        total_sum[3] + total_sum[4] + total_sum[5] +
                        total_sum[6] + total_sum[7];
      for (; k < shared_dimension; ++k) {
        final_sum += lhs[row * shared_dimension + k] *
                     rhs[k * cols_of_b + col];
      }
      result[row * cols_of_b + col] = final_sum;
    }
  }
}

void matrix_multiply_simd_blocked( const float* A, const float* B, float* C, size_t n, size_t m, size_t p) {
  const size_t block_size = 64; 
  for (size_t ii = 0; ii < n; ii += block_size) {
    for (size_t jj = 0; jj < p; jj += block_size) {
      for (size_t kk = 0; kk < m; kk += block_size) {
        for (size_t i = ii; i < std::min<std::size_t>(n, ii + block_size); ++i) {
          for (size_t j = jj; j < std::min<std::size_t>(p, jj + block_size); ++j) {
            __m256 sum = _mm256_setzero_ps();
            size_t k;
            for (k = kk; k < std::min<std::size_t>(m, kk + block_size); k += 8) {
              __m256 a = _mm256_loadu_ps(&A[i * m + k]);
              __m256 b = _mm256_loadu_ps(&B[k * p + j]);
              sum = _mm256_fmadd_ps(a, b, sum);
            }
            float total_sum[8];
            _mm256_storeu_ps(total_sum, sum);
            float final_sum = total_sum[0] + total_sum[1] + total_sum[2] +
                              total_sum[3] + total_sum[4] + total_sum[5] +
                              total_sum[6] + total_sum[7];
            for (; k < m; ++k) {
              final_sum += A[i * m + k] * B[k * p + j];
            }
            C[i * p + j] += final_sum;
          }
        }
      }
    }
  }
}

void matrix_multiply_blocked_v3(const float* A,
                                const float* B,
                                float* C,
                                size_t n, size_t m, size_t p) {
  constexpr size_t BLOCK_SIZE{4};
  for (size_t i = 0; i < n; i += BLOCK_SIZE) {
    for (size_t j = 0; j < p; j += BLOCK_SIZE) {
      for (size_t k = 0; k < m; k += BLOCK_SIZE) {
        for (size_t ii = i; ii < std::min<size_t>(i + BLOCK_SIZE, n); ++ii) {
          for (size_t jj = j; jj < std::min<size_t>(j + BLOCK_SIZE, p); ++jj) {
            float sum = 0;
            for (size_t kk = k; kk < std::min<size_t>(k + BLOCK_SIZE, m);
                 ++kk) {
              sum += A[ii * m + kk] * B[kk * p + jj];
            }
            C[ii * p + jj] += sum;
          }
        }
      }
    }
  }
}

bool is_avx2_supported() {
  int info[4];
#ifdef _MSC_VER
  __cpuidex(info, 7, 0);
#else
  __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
#endif
  return info[1] & (1 << 5);  // Check the AVX2 bit (bit 5 of ebx)
}

// Support
void matrix_multiply(const std::vector<float>& matrix_A,
                     const std::vector<float>& matrix_B,
                     std::vector<float>& matrix_C, size_t n_rows_A,
                     size_t n_cols_A_n_rows_B, size_t n_cols_B) {
  static bool checked_avx2 = false;
  static bool avx2_supported = false;

  if (!checked_avx2) {
    avx2_supported = is_avx2_supported();
    checked_avx2 = true;
    if (avx2_supported) {
      std::cout << "AVX2 is supported on this system." << std::endl;
    } else {
      std::cout << "AVX2 is not supported on this system." << std::endl;
    }
  }

  if (avx2_supported) {
    matrix_multiply_simd(matrix_A.data(), matrix_B.data(), matrix_C.data(),
                         n_rows_A, n_cols_A_n_rows_B, n_cols_B);
  } else {
    matrix_multiply_unrolled(matrix_A.data(), matrix_B.data(), matrix_C.data(),
                             n_rows_A, n_cols_A_n_rows_B, n_cols_B);
  }
}

void benchmark_naive(picobench::state& s) {
  size_t n = (int)std::sqrt(s.iterations()), m = (int)std::sqrt(s.iterations()),
         p = (int)std::sqrt(s.iterations());
  std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
  for (auto _ : s) {
    matrix_multiply_naive(A.data(), B.data(), C.data(), n, m, p);
  }
}
PICOBENCH(benchmark_naive);

void benchmark_unrolled(picobench::state& s) {
  size_t n = (int)std::sqrt(s.iterations()), m = (int)std::sqrt(s.iterations()),
         p = (int)std::sqrt(s.iterations());
  std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
  for (auto _ : s) {
    matrix_multiply_unrolled(A.data(), B.data(), C.data(), n, m, p);
  }
}
PICOBENCH(benchmark_unrolled);

void benchmark_simd(picobench::state& s) {
  size_t n = (int)std::sqrt(s.iterations()), m = (int)std::sqrt(s.iterations()),
         p = (int)std::sqrt(s.iterations());
  std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
  for (auto _ : s) {
    matrix_multiply(A, B, C, n, m, p);
  }
}
PICOBENCH(benchmark_simd);

void benchmark_blocked(picobench::state& s) {
  size_t n = (int)std::sqrt(s.iterations()), m = (int)std::sqrt(s.iterations()),
         p = (int)std::sqrt(s.iterations());
  std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
  for (auto _ : s) {
    matrix_multiply_blocked_v2(A, B, C, n, m, p);
  }
}
PICOBENCH(benchmark_blocked);

void benchmark_blocked_v3(picobench::state& s) {
  size_t n = (int)std::sqrt(s.iterations()), m = (int)std::sqrt(s.iterations()),
         p = (int)std::sqrt(s.iterations());
  std::vector<float, aligned_allocator<float>> A(n * m, 1.0f), B(m * p, 1.0f),
      C(n * p);
  for (auto _ : s) {
    matrix_multiply_blocked_v3(A.data(), B.data(), C.data(), n, m, p);
  }
}
PICOBENCH(benchmark_blocked_v3);

#if 0
void benchmark_opencl(picobench::state& s) {
    static opencl_matrix_dot cl_matrix_dot{};
    size_t n = (int)std::sqrt(s.iterations()), m = (int)std::sqrt(s.iterations()), p = (int)std::sqrt(s.iterations());
    std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
    for (auto _ : s) {
        cl_matrix_dot.multiply(A, B, C, n, m, p);
    }
}
PICOBENCH(benchmark_opencl);
#endif

void benchmark_matrix_multiply_simd_blocked(picobench::state& s) {
  size_t n = (int)std::sqrt(s.iterations()), m = (int)std::sqrt(s.iterations()),
         p = (int)std::sqrt(s.iterations());
  std::vector<float, aligned_allocator<float>> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
  for (auto _ : s) {
    matrix_multiply_simd_blocked(A.data(), B.data(), C.data(), n, m, p);
  }
}
PICOBENCH(benchmark_matrix_multiply_simd_blocked);