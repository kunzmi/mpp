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

TEST_CASE("8uC1", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AlphaPremulC]")
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
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace

    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AlphaPremulC]")
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
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_dst(size, size);
    cpu::Image<Pixel32sC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AlphaPremul(constVal.x);
    gpu_src1.AlphaPremul(constVal.x);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("16fC4A", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    cpu::Image<Pixel16fC4A> cpu_dst(size, size);
    cpu::Image<Pixel16fC4A> gpu_res(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x);
    gpu_dst.AlphaPremul(constVal.x);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AlphaPremulC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    gpu_src1.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(constVal.x, cpu_dst);
    gpu_dst.AlphaPremul(constVal.x, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AlphaPremul]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(cpu_dst);
    gpu_src1.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(cpu_dst);
    gpu_dst.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AlphaPremul]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(cpu_dst);
    gpu_src1.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(cpu_dst);
    gpu_dst.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.AlphaPremul]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(cpu_dst);
    gpu_src1.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(cpu_dst);
    gpu_dst.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AlphaPremul]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(cpu_dst);
    gpu_src1.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(cpu_dst);
    gpu_dst.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AlphaPremul]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.AlphaPremul(cpu_dst);
    gpu_src1.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.AlphaPremul(cpu_dst);
    gpu_dst.AlphaPremul(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}