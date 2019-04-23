#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main()
{
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try
        {
            program.build(devices);
        }
        catch (cl::Error const & e)
        {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        // create a message to send to kernel
        size_t const group_size = 32;
        size_t const block_size = 16;
        int N, M;
        std::cin >> N >> M;

        std::vector<double> a(N * N);
        std::vector<double> b(M * M);
        std::vector<double> c(N * N);
        for (int i = 0; i < N * N; i++) {
            std::cin >> a[i];
        }
        for (int i = 0; i < M * M; i++) {
            std::cin >> b[i];
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * N * N);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * M * M);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * N * N);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N * N, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * M * M, &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel(program, "convolution");
        kernel.setArg(0, dev_a);
        kernel.setArg(1, dev_b);
        kernel.setArg(2, dev_c);
        kernel.setArg(3, N);
        kernel.setArg(4, M);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(group_size, group_size), cl::NDRange(block_size, block_size));

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * N * N, &c[0]);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << c[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    catch (cl::Error const & e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}