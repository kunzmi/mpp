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

TEST_CASE("8uC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC1> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC1> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC1> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Integral.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size - 1, size - 1);
    cpu::Image<Pixel32sC1> cpu_dst1(size, size);
    cpu::Image<Pixel32sC1> gpu_res1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst2(size, size);
    cpu::Image<Pixel32fC1> gpu_res2(size, size);
    cpu::Image<Pixel64sC1> cpu_dst3(size, size);
    cpu::Image<Pixel64sC1> gpu_res3(size, size);
    cpu::Image<Pixel64fC1> cpu_dst4(size, size);
    cpu::Image<Pixel64fC1> gpu_res4(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size - 1, size - 1);
    gpu::Image<Pixel32sC1> gpu_dst1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst2(size, size);
    gpu::Image<Pixel64sC1> gpu_dst3(size, size);
    gpu::Image<Pixel64fC1> gpu_dst4(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Integral.UnalignedPitch]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size - 1, size - 1);
    gpu::Image<Pixel32sC1> gpu_dst1(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));

    gpu::ImageView<Pixel32sC1> gpu_dst1Pitch(gpu_dst1.Pointer(), {{size, size}, gpu_dst1.Pitch() - 1});

    CHECK_THROWS_AS(gpu_src1.Integral(gpu_dst1Pitch, 1, gpu_buffer1), PitchException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Integral.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::ImageView<Pixel32sC1> gpu_dst1(nullptr, {{size, size}, gpu_src1.Pitch()});
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));

    CHECK_THROWS_AS(gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1), NullPtrException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Integral.Roi]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));

    CHECK_THROWS_AS(gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1), RoiException);
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC2> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC2> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC2> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC3> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC3> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC3> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC4> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC4> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC4> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC1> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC1> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC1> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC2> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC2> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC2> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC3> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC3> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC3> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC4> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC4> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC4> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC1> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC1> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC1> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC2> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC2> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC2> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC3> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC3> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC3> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC4> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC4> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC4> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC1> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC1> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC1> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC2> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC2> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC2> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC3> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC3> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC3> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC4> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC4> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC4> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(1000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC1> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC1> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC1> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC2> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC2> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC2> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC3> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC3> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC3> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC4> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC4> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC4> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32uC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC1> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC1> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC1> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC1> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32uC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC2> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC2> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC2> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC2> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32uC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC3> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC3> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC3> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC3> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32uC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst1(size + 1, size + 1);
    cpu::Image<Pixel32sC4> gpu_res1(size + 1, size + 1);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64sC4> cpu_dst3(size + 1, size + 1);
    cpu::Image<Pixel64sC4> gpu_res3(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst1(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64sC4> gpu_dst3(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer1(gpu_src1.IntegralBufferSize(gpu_dst1));
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer3(gpu_src1.IntegralBufferSize(gpu_dst3));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);
    cpu_src1.Div(100000000); // reduce value range to avoid issues with 32f result

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst1, 1);
    gpu_src1.Integral(gpu_dst1, 1, gpu_buffer1);

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst3, 1);
    gpu_src1.Integral(gpu_dst3, 1, gpu_buffer3);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res1 << gpu_dst1;
    gpu_res2 << gpu_dst2;
    gpu_res3 << gpu_dst3;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst1.IsIdentical(gpu_res1));
    CHECK(cpu_dst2.IsIdentical(gpu_res2));
    CHECK(cpu_dst3.IsIdentical(gpu_res3));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.Integral.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size - 1, size - 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size, size);
    cpu::Image<Pixel32fC1> gpu_res2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst4(size, size);
    cpu::Image<Pixel64fC1> gpu_res4(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size - 1, size - 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst4(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16bfC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16bfC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16bfC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("16bfC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC1> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.Integral.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size - 1, size - 1);
    cpu::Image<Pixel32fC1> cpu_dst2(size, size);
    cpu::Image<Pixel32fC1> gpu_res2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst4(size, size);
    cpu::Image<Pixel64fC1> gpu_res4(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size - 1, size - 1);
    gpu::Image<Pixel32fC1> gpu_dst2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst4(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC2> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC3> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst2(size + 1, size + 1);
    cpu::Image<Pixel32fC4> gpu_res2(size + 1, size + 1);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst2(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer2(gpu_src1.IntegralBufferSize(gpu_dst2));
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst2, 1);
    gpu_src1.Integral(gpu_dst2, 1, gpu_buffer2);

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res2 << gpu_dst2;
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst2.IsSimilar(gpu_res2, 32));
    CHECK(cpu_dst4.IsIdentical(gpu_res4));
}

TEST_CASE("64fC1", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC1> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst4.IsSimilar(gpu_res4, 0.5));
}

TEST_CASE("64fC1", "[CUDA.Arithmetic.Integral.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size - 1, size - 1);
    cpu::Image<Pixel64fC1> cpu_dst4(size, size);
    cpu::Image<Pixel64fC1> gpu_res4(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size - 1, size - 1);
    gpu::Image<Pixel64fC1> gpu_dst4(size, size);
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst4.IsSimilar(gpu_res4, 0.5));
}

TEST_CASE("64fC2", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC2> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst4.IsSimilar(gpu_res4, 0.5));
}

TEST_CASE("64fC3", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC3> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst4.IsSimilar(gpu_res4, 0.5));
}

TEST_CASE("64fC4", "[CUDA.Arithmetic.Integral]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_dst4(size + 1, size + 1);
    cpu::Image<Pixel64fC4> gpu_res4(size + 1, size + 1);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_dst4(size + 1, size + 1);
    mpp::cuda::DevVar<byte> gpu_buffer4(gpu_src1.IntegralBufferSize(gpu_dst4));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Integral(cpu_dst4, 1);
    gpu_src1.Integral(gpu_dst4, 1, gpu_buffer4);
    gpu_res4 << gpu_dst4;

    CHECK(cpu_dst4.IsSimilar(gpu_res4, 0.5));
}