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

TEST_CASE("8uC1", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AbsDiff.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8uC1> cpu_src1(sizeUneven, sizeUneven);
    cpu::Image<Pixel8uC1> cpu_src2(sizeUneven, sizeUneven);
    cpu::Image<Pixel8uC1> cpu_dst(sizeUneven, sizeUneven);
    cpu::Image<Pixel8uC1> gpu_res(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src2(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_dst(sizeUneven, sizeUneven);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AbsDiff.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);

    gpu::ImageView<Pixel8uC1> gpu_dst(nullptr, {{size, size}, gpu_src1.Pitch()});

    CHECK_THROWS_AS(gpu_src1.AbsDiff(gpu_src2, gpu_dst), NullPtrException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AbsDiff.Roi]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    gpu_src1.SetRoi(Border(-1));

    CHECK_THROWS_AS(gpu_src1.AbsDiff(gpu_src2, gpu_dst), RoiException);
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_src2(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src2(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src2(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src2(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel8uC4A> cpu_src2(size, size);
    cpu::Image<Pixel8uC4A> cpu_dst(size, size);
    cpu::Image<Pixel8uC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src2(size, size);
    gpu::Image<Pixel8uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src2(size, size);
    gpu::Image<Pixel16uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_src2(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src2(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src2(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src2(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel16uC4A> cpu_src2(size, size);
    cpu::Image<Pixel16uC4A> cpu_dst(size, size);
    cpu::Image<Pixel16uC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src2(size, size);
    gpu::Image<Pixel16uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_src2(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src2(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_src2(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src2(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_src2(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src2(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_src2(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src2(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AbsDiff]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.AbsDiff(cpu_src2, cpu_dst);
    gpu_src1.AbsDiff(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(cpu_src2);
    gpu_src1.AbsDiff(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AbsDiffC]")
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

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC2> op(seed + 1);
    Pixel8uC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC3> op(seed + 1);
    Pixel8uC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AbsDiffC]")
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
    FillRandom<Pixel8uC4A> op(seed + 1);
    Pixel8uC4A constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AbsDiffC]")
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

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC2> op(seed + 1);
    Pixel16uC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC3> op(seed + 1);
    Pixel16uC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AbsDiffC]")
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
    FillRandom<Pixel16uC4A> op(seed + 1);
    Pixel16uC4A constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AbsDiffC]")
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

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC2> op(seed + 1);
    Pixel16fC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC3> op(seed + 1);
    Pixel16fC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC4> op(seed + 1);
    Pixel16fC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AbsDiffC]")
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

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC2> op(seed + 1);
    Pixel32fC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC3> op(seed + 1);
    Pixel32fC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AbsDiffC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AbsDiffC]")
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
    FillRandom<Pixel32fC4A> op(seed + 1);
    Pixel32fC4A constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AbsDiffDevC]")
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
    mpp::cuda::DevVar<Pixel8uC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC2> op(seed + 1);
    Pixel8uC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC3> op(seed + 1);
    Pixel8uC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AbsDiffDevC]")
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
    FillRandom<Pixel8uC4A> op(seed + 1);
    Pixel8uC4A constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC4A> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AbsDiffDevC]")
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
    mpp::cuda::DevVar<Pixel16uC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC2> op(seed + 1);
    Pixel16uC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC3> op(seed + 1);
    Pixel16uC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AbsDiffDevC]")
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
    FillRandom<Pixel16uC4A> op(seed + 1);
    Pixel16uC4A constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC4A> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AbsDiffDevC]")
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
    mpp::cuda::DevVar<Pixel16fC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC2> op(seed + 1);
    Pixel16fC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16fC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC3> op(seed + 1);
    Pixel16fC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16fC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC4> op(seed + 1);
    Pixel16fC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16fC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AbsDiffDevC]")
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
    mpp::cuda::DevVar<Pixel32fC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC2> op(seed + 1);
    Pixel32fC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC3> op(seed + 1);
    Pixel32fC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AbsDiffDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AbsDiffDevC]")
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
    FillRandom<Pixel32fC4A> op(seed + 1);
    Pixel32fC4A constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC4A> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.AbsDiff(constVal, cpu_dst);
    gpu_src1.AbsDiff(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.AbsDiff(constVal);
    gpu_src1.AbsDiff(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}