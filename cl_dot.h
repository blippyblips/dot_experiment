#pragma once

#pragma comment(lib, "OpenCL.lib")

#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <fstream>

class opencl_matrix_dot {
public:
    opencl_matrix_dot() {
        cl_int err;

        // Get platform and device information
        cl::Platform::get(&platforms);
        platform = platforms.front();

        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        device = devices.front();

        // Create an OpenCL context
        context = cl::Context(device);

        // Create command queue
        queue = cl::CommandQueue(context, device);

        // Load and build OpenCL kernel
        kernel_code = load_kernel("matrix_multiply_kernel.cl");
        program = cl::Program(context, kernel_code);
        err = program.build(devices);

        if (err != CL_SUCCESS) {
            std::cerr << "Error building the program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        }
    }

    ~opencl_matrix_dot() {
        // Destructor will be called to release resources
    }

    void multiply(const std::vector<float>& A,
                  const std::vector<float>& B,
                  std::vector<float>& C,
                  size_t n, size_t m, size_t p) {
        // Create buffers
        cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * n * m);
        cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, sizeof(float) * m * p);
        cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * p);

        // Copy data to buffers
        queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * n * m, A.data());
        queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * m * p, B.data());

        // Set kernel arguments
        cl::Kernel kernel(program, "matrix_multiply");
        kernel.setArg(0, buffer_A);
        kernel.setArg(1, buffer_B);
        kernel.setArg(2, buffer_C);
        kernel.setArg(3, static_cast<int>(m));

        // Execute the kernel
        cl::NDRange global_size(n, p);
        cl::NDRange local_size(16, 16);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size);

        // Read the result
        queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * n * p, C.data());
    }

private:
    std::vector<cl::Platform> platforms;
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    std::string kernel_code;
    cl::Program program;

    std::string load_kernel(const std::string& path) {
        std::ifstream file(path);
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return content;
    }
};
