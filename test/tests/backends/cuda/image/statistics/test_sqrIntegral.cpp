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

constexpr int size = 256;

TEST_CASE("8uC1", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC1> cpu_dst2Sqr(size + 1, size + 1);
    cpu::Image<Pixel64sC1> gpu_res2Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC1> gpu_dst1Sqr(size + 1, size + 1);
    gpu::Image<Pixel32sC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC1> gpu_dst2Sqr(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst3Sqr(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4Sqr(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.SqrIntegralBufferSize(gpu_dst2, gpu_dst2Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.SqrIntegralBufferSize(gpu_dst3, gpu_dst3Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.SqrIntegralBufferSize(gpu_dst4, gpu_dst4Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_src1.SqrIntegral(cpu_dst2, cpu_dst2Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst2, gpu_dst2Sqr, 1, 1, gpu_buffer2);

    cpu_src1.SqrIntegral(cpu_dst3, cpu_dst3Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst3, gpu_dst3Sqr, 1, 1, gpu_buffer3);

    cpu_src1.SqrIntegral(cpu_dst4, cpu_dst4Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst4, gpu_dst4Sqr, 1, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;
    gpu_res1Sqr << gpu_dst1Sqr;
    gpu_res2Sqr << gpu_dst2Sqr;
    gpu_res3Sqr << gpu_dst3Sqr;
    gpu_res4Sqr << gpu_dst4Sqr;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
    CHECK(cpu_dst1Sqr.IsIdentical(gpu_res1Sqr));
    CHECK(cpu_dst2Sqr.IsIdentical(gpu_res2Sqr));
    CHECK(cpu_dst3Sqr.IsIdentical(gpu_res3Sqr));
    CHECK(cpu_dst4Sqr.IsIdentical(gpu_res4Sqr));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.SqrIntegral.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size - 1, size - 1);
    cpu::Image<Pixel32sC1> cpu_dst1(size, size);
    cpu::Image<Pixel32sC1> gpu_res1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1Sqr(size, size);
    cpu::Image<Pixel32sC1> gpu_res1Sqr(size, size);
    cpu::Image<Pixel32sC1> cpu_dst2(size, size);
    cpu::Image<Pixel32sC1> gpu_res2(size, size);
    cpu::Image<Pixel64sC1> cpu_dst2Sqr(size, size);
    cpu::Image<Pixel64sC1> gpu_res2Sqr(size, size);
    cpu::Image<Pixel32fC1> cpu_dst3(size, size);
    cpu::Image<Pixel32fC1> gpu_res3(size, size);
    cpu::Image<Pixel64fC1> cpu_dst3Sqr(size, size);
    cpu::Image<Pixel64fC1> gpu_res3Sqr(size, size);
    cpu::Image<Pixel64fC1> cpu_dst4(size, size);
    cpu::Image<Pixel64fC1> gpu_res4(size, size);
    cpu::Image<Pixel64fC1> cpu_dst4Sqr(size, size);
    cpu::Image<Pixel64fC1> gpu_res4Sqr(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size - 1, size - 1);
    gpu::Image<Pixel32sC1> gpu_dst1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1Sqr(size, size);
    gpu::Image<Pixel32sC1> gpu_dst2(size, size);
    gpu::Image<Pixel64sC1> gpu_dst2Sqr(size, size);
    gpu::Image<Pixel32fC1> gpu_dst3(size, size);
    gpu::Image<Pixel64fC1> gpu_dst3Sqr(size, size);
    gpu::Image<Pixel64fC1> gpu_dst4(size, size);
    gpu::Image<Pixel64fC1> gpu_dst4Sqr(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.SqrIntegralBufferSize(gpu_dst2, gpu_dst2Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.SqrIntegralBufferSize(gpu_dst3, gpu_dst3Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.SqrIntegralBufferSize(gpu_dst4, gpu_dst4Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_src1.SqrIntegral(cpu_dst2, cpu_dst2Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst2, gpu_dst2Sqr, 1, 1, gpu_buffer2);

    cpu_src1.SqrIntegral(cpu_dst3, cpu_dst3Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst3, gpu_dst3Sqr, 1, 1, gpu_buffer3);

    cpu_src1.SqrIntegral(cpu_dst4, cpu_dst4Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst4, gpu_dst4Sqr, 1, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;
    gpu_res1Sqr << gpu_dst1Sqr;
    gpu_res2Sqr << gpu_dst2Sqr;
    gpu_res3Sqr << gpu_dst3Sqr;
    gpu_res4Sqr << gpu_dst4Sqr;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
    CHECK(cpu_dst1Sqr.IsIdentical(gpu_res1Sqr));
    CHECK(cpu_dst2Sqr.IsIdentical(gpu_res2Sqr));
    CHECK(cpu_dst3Sqr.IsIdentical(gpu_res3Sqr));
    CHECK(cpu_dst4Sqr.IsIdentical(gpu_res4Sqr));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.SqrIntegral.UnalignedPitch]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size - 1, size - 1);
    gpu::Image<Pixel32sC1> gpu_dst1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst2(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1));

    gpu::ImageView<Pixel32sC1> gpu_dst1Pitch(gpu_dst1.Pointer(), {{size, size}, gpu_dst1.Pitch() - 1});

    CHECK_THROWS_AS(gpu_src1.SqrIntegral(gpu_dst1Pitch, gpu_dst2, 1, 1, gpu_buffer1), PitchException);
    CHECK_THROWS_AS(gpu_src1.SqrIntegral(gpu_dst2, gpu_dst1Pitch, 1, 1, gpu_buffer1), PitchException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.SqrIntegral.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::ImageView<Pixel32sC1> gpu_dst1(nullptr, {{size, size}, gpu_src1.Pitch()});
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1));

    CHECK_THROWS_AS(gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1, 1, 1, gpu_buffer1), NullPtrException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.SqrIntegral.Roi]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1));

    CHECK_THROWS_AS(gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1, 1, 1, gpu_buffer1), RoiException);
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC2> cpu_dst2Sqr(size + 1, size + 1);
    cpu::Image<Pixel64sC2> gpu_res2Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC2> gpu_dst1Sqr(size + 1, size + 1);
    gpu::Image<Pixel32sC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC2> gpu_dst2Sqr(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst3Sqr(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4Sqr(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.SqrIntegralBufferSize(gpu_dst2, gpu_dst2Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.SqrIntegralBufferSize(gpu_dst3, gpu_dst3Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.SqrIntegralBufferSize(gpu_dst4, gpu_dst4Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_src1.SqrIntegral(cpu_dst2, cpu_dst2Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst2, gpu_dst2Sqr, 1, 1, gpu_buffer2);

    cpu_src1.SqrIntegral(cpu_dst3, cpu_dst3Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst3, gpu_dst3Sqr, 1, 1, gpu_buffer3);

    cpu_src1.SqrIntegral(cpu_dst4, cpu_dst4Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst4, gpu_dst4Sqr, 1, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;
    gpu_res1Sqr << gpu_dst1Sqr;
    gpu_res2Sqr << gpu_dst2Sqr;
    gpu_res3Sqr << gpu_dst3Sqr;
    gpu_res4Sqr << gpu_dst4Sqr;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
    CHECK(cpu_dst1Sqr.IsIdentical(gpu_res1Sqr));
    CHECK(cpu_dst2Sqr.IsIdentical(gpu_res2Sqr));
    CHECK(cpu_dst3Sqr.IsIdentical(gpu_res3Sqr));
    CHECK(cpu_dst4Sqr.IsIdentical(gpu_res4Sqr));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC3> cpu_dst2Sqr(size + 1, size + 1);
    cpu::Image<Pixel64sC3> gpu_res2Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC3> gpu_dst1Sqr(size + 1, size + 1);
    gpu::Image<Pixel32sC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC3> gpu_dst2Sqr(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst3Sqr(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4Sqr(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.SqrIntegralBufferSize(gpu_dst2, gpu_dst2Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.SqrIntegralBufferSize(gpu_dst3, gpu_dst3Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.SqrIntegralBufferSize(gpu_dst4, gpu_dst4Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_src1.SqrIntegral(cpu_dst2, cpu_dst2Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst2, gpu_dst2Sqr, 1, 1, gpu_buffer2);

    cpu_src1.SqrIntegral(cpu_dst3, cpu_dst3Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst3, gpu_dst3Sqr, 1, 1, gpu_buffer3);

    cpu_src1.SqrIntegral(cpu_dst4, cpu_dst4Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst4, gpu_dst4Sqr, 1, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;
    gpu_res1Sqr << gpu_dst1Sqr;
    gpu_res2Sqr << gpu_dst2Sqr;
    gpu_res3Sqr << gpu_dst3Sqr;
    gpu_res4Sqr << gpu_dst4Sqr;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
    CHECK(cpu_dst1Sqr.IsIdentical(gpu_res1Sqr));
    CHECK(cpu_dst2Sqr.IsIdentical(gpu_res2Sqr));
    CHECK(cpu_dst3Sqr.IsIdentical(gpu_res3Sqr));
    CHECK(cpu_dst4Sqr.IsIdentical(gpu_res4Sqr));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.SqrIntegral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> cpu_dst1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1Sqr(size + 1, size + 1);
    cpu::Image<Pixel32sC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC4> cpu_dst2Sqr(size + 1, size + 1);
    cpu::Image<Pixel64sC4> gpu_res2Sqr(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res3Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4Sqr(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4Sqr(size + 1, size + 1);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32sC4> gpu_dst1Sqr(size + 1, size + 1);
    gpu::Image<Pixel32sC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC4> gpu_dst2Sqr(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst3Sqr(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4Sqr(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.SqrIntegralBufferSize(gpu_dst1, gpu_dst1Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.SqrIntegralBufferSize(gpu_dst2, gpu_dst2Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.SqrIntegralBufferSize(gpu_dst3, gpu_dst3Sqr));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.SqrIntegralBufferSize(gpu_dst4, gpu_dst4Sqr));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.SqrIntegral(cpu_dst1, cpu_dst1Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst1, gpu_dst1Sqr, 1, 1, gpu_buffer1);

    cpu_src1.SqrIntegral(cpu_dst2, cpu_dst2Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst2, gpu_dst2Sqr, 1, 1, gpu_buffer2);

    cpu_src1.SqrIntegral(cpu_dst3, cpu_dst3Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst3, gpu_dst3Sqr, 1, 1, gpu_buffer3);

    cpu_src1.SqrIntegral(cpu_dst4, cpu_dst4Sqr, 1, 1);
    gpu_src1.SqrIntegral(gpu_dst4, gpu_dst4Sqr, 1, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;
    gpu_res1Sqr << gpu_dst1Sqr;
    gpu_res2Sqr << gpu_dst2Sqr;
    gpu_res3Sqr << gpu_dst3Sqr;
    gpu_res4Sqr << gpu_dst4Sqr;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
    CHECK(cpu_dst1Sqr.IsIdentical(gpu_res1Sqr));
    CHECK(cpu_dst2Sqr.IsIdentical(gpu_res2Sqr));
    CHECK(cpu_dst3Sqr.IsIdentical(gpu_res3Sqr));
    CHECK(cpu_dst4Sqr.IsIdentical(gpu_res4Sqr));
}