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

TEST_CASE("8sC1", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel8sC1> cpu_dst(size, size);
    cpu::Image<Pixel8sC1> gpu_res(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel8sC2> cpu_dst(size, size);
    cpu::Image<Pixel8sC2> gpu_res(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel8sC3> cpu_dst(size, size);
    cpu::Image<Pixel8sC3> gpu_res(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_dst(size, size);
    cpu::Image<Pixel8sC4> gpu_res(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC4A", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    cpu::Image<Pixel8sC4A> cpu_dst(size, size);
    cpu::Image<Pixel8sC4A> gpu_res(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel8uC4A> cpu_dst(size, size);
    cpu::Image<Pixel8uC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 8;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> gpu_res(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel16sC2> cpu_dst(size, size);
    cpu::Image<Pixel16sC2> gpu_res(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> gpu_res(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> gpu_res(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC4A", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    cpu::Image<Pixel16sC4A> cpu_dst(size, size);
    cpu::Image<Pixel16sC4A> gpu_res(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel16uC4A> cpu_dst(size, size);
    cpu::Image<Pixel16uC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 16;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_dst(size, size);
    cpu::Image<Pixel32sC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC1", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel32uC1> cpu_dst(size, size);
    cpu::Image<Pixel32uC1> gpu_res(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC2", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel32uC2> cpu_dst(size, size);
    cpu::Image<Pixel32uC2> gpu_res(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC3", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel32uC3> cpu_dst(size, size);
    cpu::Image<Pixel32uC3> gpu_res(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC4", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel32uC4> cpu_dst(size, size);
    cpu::Image<Pixel32uC4> gpu_res(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC4A", "[CUDA.Arithmetic.LShift]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32uC4A> cpu_dst(size, size);
    cpu::Image<Pixel32uC4A> gpu_res(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 constVal;
    op(constVal);
    constVal.x %= 32;

    cpu_src1 >> gpu_src1;

    cpu_src1.LShift(constVal.x, cpu_dst);
    gpu_src1.LShift(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.LShift(constVal.x);
    gpu_src1.LShift(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}