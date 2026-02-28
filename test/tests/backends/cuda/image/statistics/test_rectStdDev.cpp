// on windows the dllimport flag prohibits the instantiation of cpu::Image<Pixel64sC1>
// as it is not exported in any DLL. To "fix" this, we simply undef the export macro for
// imageView and re-define it as an empty macro:
#ifdef _WIN32
#include <backends/cuda/image/dllexport_cudai.h>
#undef MPPEXPORT_CUDAI
#define MPPEXPORT_CUDAI
#endif

#include <backends/cuda/devVar.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/pitchException.h>
#include <common/image/roiException.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace gpu = mpp::image::cuda;

constexpr int size       = 256;
constexpr int filterSize = 11;

TEST_CASE("8uC1", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC1> gpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dstStd1(size, size);
    cpu::Image<Pixel32fC1> gpu_resStd1(size, size);
    gpu::Image<Pixel32fC1> gpu_dstStd1(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_resStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));

    cpu_dst1.RectStdDev(cpu_dst1Sqr, cpu_dstStd1, FilterArea(filterSize, 0));
    gpu_dst1.RectStdDev(gpu_dst1Sqr, gpu_dstStd1, FilterArea(filterSize, 0));

    gpu_resStd1 << gpu_dstStd1;

    CHECK(cpu_dstStd1.IsIdentical(gpu_resStd1));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC2> gpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dstStd1(size, size);
    cpu::Image<Pixel32fC2> gpu_resStd1(size, size);
    gpu::Image<Pixel32fC2> gpu_dstStd1(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_resStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));

    cpu_dst1.RectStdDev(cpu_dst1Sqr, cpu_dstStd1, FilterArea(filterSize, 0));
    gpu_dst1.RectStdDev(gpu_dst1Sqr, gpu_dstStd1, FilterArea(filterSize, 0));

    gpu_resStd1 << gpu_dstStd1;

    CHECK(cpu_dstStd1.IsIdentical(gpu_resStd1));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC3> gpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dstStd1(size, size);
    cpu::Image<Pixel32fC3> gpu_resStd1(size, size);
    gpu::Image<Pixel32fC3> gpu_dstStd1(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_resStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));

    cpu_dst1.RectStdDev(cpu_dst1Sqr, cpu_dstStd1, FilterArea(filterSize, 0));
    gpu_dst1.RectStdDev(gpu_dst1Sqr, gpu_dstStd1, FilterArea(filterSize, 0));

    gpu_resStd1 << gpu_dstStd1;

    CHECK(cpu_dstStd1.IsIdentical(gpu_resStd1));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC4> gpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dstStd1(size, size);
    cpu::Image<Pixel32fC4> gpu_resStd1(size, size);
    gpu::Image<Pixel32fC4> gpu_dstStd1(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dst1Sqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_dstStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    gpu_resStd1.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));

    cpu_dst1.RectStdDev(cpu_dst1Sqr, cpu_dstStd1, FilterArea(filterSize, 0));
    gpu_dst1.RectStdDev(gpu_dst1Sqr, gpu_dstStd1, FilterArea(filterSize, 0));

    gpu_resStd1 << gpu_dstStd1;

    CHECK(cpu_dstStd1.IsIdentical(gpu_resStd1));
}
