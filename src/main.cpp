
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
#include <common/needSaurationClamp.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>

using namespace opp;
using namespace opp::image;

template <size_t tupelSize, typename ComputeType, typename ResultType, typename SrcType, typename operation>
struct TestFunctor
{
    const SrcType *Src1;
    size_t SrcPitch1;

    const SrcType *Src2;
    size_t SrcPitch2;

    operation Op;
    float ScaleFactor;

    TestFunctor()
    {
    }

    TestFunctor(SrcType *aSrc1, size_t aSrcPitch1, SrcType *aSrc2, size_t aSrcPitch2, operation aOp, float aScaleFactor)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Src2(aSrc2), SrcPitch2(aSrcPitch2), Op(aOp), ScaleFactor(aScaleFactor)
    {
    }

    void operator()(int aPixelX, int aPixelY, ResultType &aResult)
        requires std::integral<typename remove_vector<ResultType>::type> &&
                 std::floating_point<typename remove_vector<ComputeType>::type>
    {
        if (aPixelX > aPixelY)
        {
            return;
        }
        const SrcType *pixelSrc1 = Src1;
        const SrcType *pixelSrc2 = Src2;
        ComputeType temp;
        Op(ComputeType(*pixelSrc1), ComputeType(*pixelSrc2), temp);
        temp *= ScaleFactor;
        // temp.ClampToTargetType(aResult);
        temp.ClampToTargetType<typename remove_vector<ResultType>::type>();
        aResult = ResultType(temp);
    }
};

template <VectorType T> int check(T hallo)
{
    return hallo.x;
}

void forEachPixelKernelWithCuda(const int *aIn1, size_t aPitch1, const int *aIn2, size_t aPitch2, int *aOut,
                                size_t aPitchOut, Size2D aSize);
int main()
{
    try
    {

        Add<Vector1<float>> op;
        Vector1<int> in1(1000);
        Vector1<int> in2(10);
        Vector1<int> outres;
        Vector1<byte> outresByte(in1);
        byte tt(static_cast<byte>(in1.x));
        if (tt)
        {
        }

        TestFunctor<2, Vector1<float>, Vector1<int>, Vector1<int>, Add<Vector1<float>>> functor(&in1, 1, &in2, 1, op,
                                                                                                1.0f);
        functor(0, 0, outres);

        Vector1<float> ttt(1024);
        ttt.ClampToTargetType<Vector1<byte>>();

        Magnitude<Complex<double>> testMag;
        Complex<double> testVal1(10, -10);
        Complex<short> testVal2(-10, 100);
        Complex<short> testVal3;
        double res = 0;
        testMag(testVal1, res);

        const Size2D imgSize{1025, 1024};

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        std::vector<int> vecImg1(imgSize.TotalSize());
        std::vector<int> vecImg2(imgSize.TotalSize());
        std::vector<int> vecImg3(imgSize.TotalSize());

        int *img1 = vecImg1.data();
        int *img2 = vecImg2.data();
        int *out  = vecImg3.data();

        for (size_t i = 0; i < imgSize.TotalSize(); i++)
        {
            img1[i] = to_int(i);
            img2[i] = 2;
            out[i]  = 0;
        }

        int *dev_in1 = nullptr;
        int *dev_in2 = nullptr;
        int *dev_out = nullptr;

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
        cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&dev_in1), &pitch1, sizeof(int) * to_size_t(imgSize.x),
                                     to_size_t(imgSize.y));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed!";
            return 1;
        }

        cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&dev_in2), &pitch2, sizeof(int) * to_size_t(imgSize.x),
                                     to_size_t(imgSize.y));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed!";
            return 1;
        }

        cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&dev_out), &pitchOut, sizeof(int) * to_size_t(imgSize.x),
                                     to_size_t(imgSize.y));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed!";
            return 1;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy2D(dev_in1, pitch1, img1, sizeof(int) * to_size_t(imgSize.x),
                                  sizeof(int) * to_size_t(imgSize.x), to_size_t(imgSize.y), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!";
            return 1;
        }

        cudaStatus = cudaMemcpy2D(dev_in2, pitch2, img2, sizeof(int) * to_size_t(imgSize.x),
                                  sizeof(int) * to_size_t(imgSize.x), to_size_t(imgSize.y), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!";
            return 1;
        }

        cudaStatus = cudaMemcpy2D(dev_out, pitchOut, out, sizeof(int) * to_size_t(imgSize.x),
                                  sizeof(int) * to_size_t(imgSize.x), to_size_t(imgSize.y), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!";
            return 1;
        }

        forEachPixelKernelWithCuda(dev_in1, pitch1, dev_in2, pitch2, dev_out, pitchOut, imgSize);

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
            if (out[i] != img1[i] + img2[i])
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
