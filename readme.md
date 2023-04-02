# A study in dot (Optimizing Matrix Dot Product)

    ## 1. Intro

    ## 2. Optimizing
    ### 2.1 Naive
    ### 2.2 Loop Unrolling
    ### 2.3 AVX2/SIMD Instructions
    ### 2.4 Cache Blocking
    ### 2.5 OpenCL

    ## 3. Benchmarks

    ## 4. References
---

## 1. Intro

Needed an optimized version of the dot product for a project and ended up putting together this doc a long the way.
I'll probably come back to this at some point and add stats and more optimizations/alternative algorithms but for the
purposes of my project, this has reached the point where i got what i need for now.

---

# 2. Optimizing

## 2.1 Naive Implementation (O(n^3))
The naive implementation uses three nested loops to compute the matrix product.

```cpp
void matrix_multiply_naive(const float* A, const float* B, float* __restrict C, size_t n, size_t m, size_t p) {
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
```

---

## 2.2 Loop Unrolling

Loop unrolling reduces the number of iterations in a loop by executing multiple iterations in a single loop cycle.

```cpp
void matrix_multiply_unrolled(const float* A, const float* B, float* __restrict C, size_t n, size_t m, size_t p) {
  constexpr size_t unroll_factor = 4;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      std::array<float, unroll_factor> sum{};
      std::memset(sum.data(), 0, sum.size());

      size_t k;
      for (k = 0; k + unroll_factor - 1 < m; k += unroll_factor) {
          for (size_t u = 0; u < unroll_factor; ++u) {
              sum[u] += A[i * m + k + u] * B[(k + u) * p + j];
          }
      }
      
      float total_sum = 0.0f;
      for(size_t s = 0; s < unroll_factor; ++s) {
        total_sum += sum[s];
      }
      for (; k < m; ++k) {
          total_sum += A[i * m + k] * B[k * p + j];
      }
      C[i * p + j] = total_sum;
    }
  }
}
```

---

## 2.3 Vectorizing

```cpp
void matrix_multiply_simd(const float* lhs, const float* rhs, float* result, size_t rows_of_a, size_t shared_dimension, size_t cols_of_b) {
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
```

---

## 2.4 Cache Blocking

Cache blocking improves cache locality by dividing matrices into smaller "blocks" to reduce cache misses and improve performance.

```cpp
void matrix_multiply_blocked(const float* A, const float* B, float* __restrict C, size_t n, size_t m, size_t p) {
  constexpr size_t BLOCK_SIZE{64};
  for (size_t i = 0; i < n; i += BLOCK_SIZE) {
    for (size_t j = 0; j < p; j += BLOCK_SIZE) {
      for (size_t k = 0; k < m; k += BLOCK_SIZE) {
        for (size_t ii = i; ii < std::min(i + BLOCK_SIZE, n); ++ii) {
          for (size_t jj = j; jj < std::min(j + BLOCK_SIZE, p); ++jj) {
            float sum = 0;
            for (size_t kk = k; kk < std::min(k + BLOCK_SIZE, m); ++kk) {
                sum += A[ii * m + kk] * B[kk * p + jj];
            }
            C[ii * p + jj] += sum;
          }
        }
      }
    }
  }
}
```

---

## 2.5 OpenCL

Used OpenCL to keep things entirely in c++ without need for specialized compiler.
However with NVCC(NVidia Cuda Compiler) catching up to the c++ standard you can sort of do regular c++ with it and combining with NVidia Thrust
it can work a lot like coding against STL so might be something to investigate soon.

```cpp
struct opencl_context {
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;

  opencl_context() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    device = devices.front();

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    auto content = std::string((std::istreambuf_iterator<char>(std::ifstream("matrix_multiply_kernel.cl"))), std::istreambuf_iterator<char>());
    cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.length()));
    cl::Program program(context, sources);

    if (program.build(devices) != CL_SUCCESS) {
        std::cerr << "Error building the program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return;
    }
  }

  static opencl_context instance() {
    static opencl_context _instance{};
    return _instance;
  }
};

void matrix_multiply_opencl(const float* A, const float* B, float* C, size_t n, size_t m, size_t p) {
    auto opencl = opencl_context::instance();    

    auto buffer_A = cl::Buffer(opencl.context, CL_MEM_READ_ONLY, sizeof(float) * n * m);
    auto buffer_B = cl::Buffer(opencl.context, CL_MEM_READ_ONLY, sizeof(float) * m * p);
    auto buffer_C = cl::Buffer(opencl.context, CL_MEM_WRITE_ONLY, sizeof(float) * n * p);

    opencl.queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * n * m, A.data());
    opencl.queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * m * p, B.data());

    auto kernel = cl::Kernel(program, "matrix_multiply");
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    kernel.setArg(2, buffer_C);
    kernel.setArg(3, static_cast<int>(m));

    cl::NDRange global_size(n, p);
    cl::NDRange local_size(16, 16);
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size);
    opencl.queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * n * p, C.data());
}
```

```cl
matrix_multiply_kernel.cl:
__kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C, int m) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < m; ++k) {
        sum += A[i * m + k] * B[k * get_global_size(1) + j];
    }

    C[i * get_global_size(1) + j] = sum;
}
```

---

# 3. Microbenchmarks

```cpp
#include <picobench/picobench.hpp>

size_t n = 256, m = 256, p = 256; // can use s.iterations() in the benchmarks to scale the tests by dimension instead of only by iteration

void benchmark_naive(picobench::state& s) {
    std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
    for (auto _ : s) {
        matrix_multiply_naive(A, B, C, n, m, p);
    }
}
PICOBENCH(benchmark_naive);

void benchmark_unrolled(picobench::state& s) {
    std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
    for (auto _ : s) {
        matrix_multiply_unrolled(A, B, C, n, m, p);
    }
}
PICOBENCH(benchmark_unrolled);

void benchmark_simd(picobench::state& s) {
    std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
    for (auto _ : s) {
        matrix_multiply_simd(A, B, C, n, m, p);
    }
}
PICOBENCH(benchmark_simd);

void benchmark_opencl(picobench::state& s) {
    std::vector<float> A(n * m, 1.0f), B(m * p, 1.0f), C(n * p);
    for (auto _ : s) {
        matrix_multiply_opencl(A, B, C, n, m, p);
    }
}
PICOBENCH(benchmark_opencl);
```

Don't forget to enable AVX2 support in the compiler flags.

---

## 4. References and sources
https://en.algorithmica.org/hpc/algorithms/matmul/ (Huge thanks to algorithmica, inspired me to produce something similar for dot product)
https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
https://www.cs.utexas.edu/users/pingali/CS378/2008sp/papers/gotoPaper.pdf