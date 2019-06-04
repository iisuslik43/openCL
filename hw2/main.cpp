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
        std::ifstream cl_file("scan.cl");
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
            std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cerr << log_str;
            return 0;
        }

        // create a message to send to kernel
        size_t const block_size = 256;
        int N;
        std::cin >> N;
        int GPU_N = (N / block_size + 1) * block_size;
        int blocks = N / block_size + 1;

        std::vector<double> input(GPU_N);
        std::vector<double> output(GPU_N);
        std::vector<double> sum(blocks);
        for (int i = 0; i < N; i++) {
            std::cin >> input[i];
        }

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * GPU_N);
        cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(double) * GPU_N);
        cl::Buffer dev_sum(context, CL_MEM_READ_WRITE, sizeof(double) * blocks);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * GPU_N, &input[0]);
        queue.enqueueWriteBuffer(dev_output, CL_TRUE, 0, sizeof(double) * GPU_N, &output[0]);
        queue.enqueueWriteBuffer(dev_sum, CL_TRUE, 0, sizeof(double) * blocks, &sum[0]);

        // load named kernel from opencl source
        cl::Kernel kernel1(program, "scan");
        kernel1.setArg(0, dev_input);
        kernel1.setArg(1, dev_output);
        kernel1.setArg(2, dev_sum);
        kernel1.setArg(3, cl::__local(sizeof(double) * block_size));
        kernel1.setArg(4, cl::__local(sizeof(double) * block_size));
        kernel1.setArg(5, N);
        kernel1.setArg(6, GPU_N);
        queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(GPU_N), cl::NDRange(block_size));

        queue.enqueueReadBuffer(dev_sum, CL_TRUE, 0, sizeof(double) * blocks, &sum[0]);

        for (int i = 0; i < blocks; i++) {
            double prev = i == 0 ? 0 : sum[i - 1];
            sum[i] = sum[i] + prev;
        }
        queue.enqueueWriteBuffer(dev_sum, CL_TRUE, 0, sizeof(double) * blocks, &sum[0]);

        // load named kernel from opencl source
        cl::Kernel kernel2(program, "add_sum");
        kernel2.setArg(0, dev_output);
        kernel2.setArg(1, dev_sum);
        kernel2.setArg(2, N);
        kernel2.setArg(3, GPU_N);
        queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(GPU_N), cl::NDRange(block_size));

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * GPU_N, &output[0]);

        for (int i = 0; i < N; i++) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
    }
    catch (cl::Error const & e)
    {
        std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}