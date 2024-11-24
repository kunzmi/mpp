
#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <common/version.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector>

#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/ternary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/image/pixelTypes.h>
#include <common/vector_typetraits.h>
#include <common/vector4A.h>
#include <common/vectorTypes.h>

#include <backends/cuda/image/addKernel.h>
#include <backends/cuda/image/configurations.h>

#include <backends/cuda/streamCtx.h>

#include <common/scratchBuffer.h>

using namespace opp;
using namespace opp::image;
using namespace opp::cuda::image;

int main()
{
    try
    {
        std::array<size_t, 3> sizes{1000, 200, 64};
        ScratchBuffer<byte, int, double> buffer(nullptr, sizes);

        auto ptr1 = buffer.Get<0>();
        auto ptr2 = buffer.Get<1>();
        auto ptr3 = buffer.Get<2>();

        size_t size1 = buffer.GetSubBufferSize(0);
        size_t size2 = buffer.GetSubBufferSize(1);
        size_t size3 = buffer.GetSubBufferSize(2);

        size_t sizeTotal = buffer.GetTotalBufferSize();

        if (size1 + size2 + size3 == sizeTotal)
        {
            if (ptr1 != nullptr && ptr2 != nullptr && ptr3 != nullptr)
            {
            }
        }

        Vector4<int> vec4i(1, 6, 3, 7);
        Vector4<int> alpha(5, 6, 7, 8);

        byte eqt = vec4i < alpha;
        if (eqt)
        {
        }

        opp::cuda::StreamCtx ctx = opp::cuda::StreamCtxSingleton::Get();
        std::cout << ctx.DeviceId;

        const Size2D imgSize{1025, 1024};

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        std::vector<Pixel8uC4A> vecImg1(imgSize.TotalSize());
        std::vector<Pixel8uC4A> vecImg2(imgSize.TotalSize());
        std::vector<Pixel8uC4A> vecImg3(imgSize.TotalSize());

        Pixel8uC4A *img1 = vecImg1.data();
        Pixel8uC4A *img2 = vecImg2.data();
        Pixel8uC4A *out  = vecImg3.data();

        for (size_t i = 0; i < imgSize.TotalSize(); i++)
        {
            img1[i]  = Pixel8uC4A(to_byte(i % 255));
            img2[i]  = Pixel8uC4A(2);
            out[i]   = Pixel8uC4A(to_byte(0));
            out[i].w = 128;
        }

        Pixel8uC4A *dev_in1 = nullptr;
        Pixel8uC4A *dev_in2 = nullptr;
        Pixel8uC4A *dev_out = nullptr;

        size_t pitch1          = 0;
        size_t pitch2          = 0;
        size_t pitchOut        = 0;
        cudaError_t cudaStatus = cudaSuccess;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
            return 1;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&dev_in1), &pitch1,
                                     sizeof(Pixel8uC4) * to_size_t(imgSize.x), to_size_t(imgSize.y));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed!";
            return 1;
        }

        cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&dev_in2), &pitch2,
                                     sizeof(Pixel8uC4) * to_size_t(imgSize.x), to_size_t(imgSize.y));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed!";
            return 1;
        }

        cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&dev_out), &pitchOut,
                                     sizeof(Pixel8uC4) * to_size_t(imgSize.x), to_size_t(imgSize.y));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed!";
            return 1;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy2D(dev_in1, pitch1, img1, sizeof(Pixel8uC4) * to_size_t(imgSize.x),
                                  sizeof(int) * to_size_t(imgSize.x), to_size_t(imgSize.y), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!";
            return 1;
        }

        cudaStatus = cudaMemcpy2D(dev_in2, pitch2, img2, sizeof(Pixel8uC4) * to_size_t(imgSize.x),
                                  sizeof(int) * to_size_t(imgSize.x), to_size_t(imgSize.y), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!";
            return 1;
        }

        cudaStatus = cudaMemcpy2D(dev_out, pitchOut, out, sizeof(Pixel8uC4) * to_size_t(imgSize.x),
                                  sizeof(int) * to_size_t(imgSize.x), to_size_t(imgSize.y), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!";
            return 1;
        }

        InvokeAddSrcSrc(dev_in1, pitch1, dev_in2, pitch2, dev_out, pitchOut, imgSize);

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy2D(out, sizeof(int) * to_size_t(imgSize.x), dev_out, pitchOut,
                                  sizeof(int) * to_size_t(imgSize.x), to_size_t(imgSize.y), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!";
            return 1;
        }

        int check_host = 0;
        for (size_t i = 0; i < imgSize.TotalSize(); i++)
        {
            Pixel8uC4A temp = Pixel8uC4A(Pixel32sC4A(img1[i]) + Pixel32sC4A(img2[i]));

            if (out[i] != temp)
            {
                std::cout << "Wrong result in pixel " << imgSize.GetCoordinates(i) << std::endl;
                check_host++;
            }
        }

        std::cout << "Number of wrong pixels: " << check_host << std::endl;
    }
    catch (...)
    {
        return 1;
    }
    return 0;
}
