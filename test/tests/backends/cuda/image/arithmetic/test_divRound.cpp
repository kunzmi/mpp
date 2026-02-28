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

constexpr int size = 128;

TEST_CASE("8uC1", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    gpu_src1 >> gpu_dst;
    cpu_src1 >> cpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.DivRound]")
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

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.DivRound]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_src2(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src2(size, size);
    gpu::Image<Pixel32sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.DivRound]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_src2(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src2(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.DivRound]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_src2(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src2(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.DivRound]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_src2(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_src2(size, size);
    gpu::Image<Pixel16scC1> gpu_dst(size, size);
    Pixel64fC1 meanError;

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.DivRound]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_src2(size, size);
    cpu::Image<Pixel16scC2> cpu_dst(size, size);
    cpu::Image<Pixel16scC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_src2(size, size);
    gpu::Image<Pixel16scC2> gpu_dst(size, size);
    Pixel64fC2 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.DivRound]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_src2(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_src2(size, size);
    gpu::Image<Pixel16scC3> gpu_dst(size, size);
    Pixel64fC3 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.DivRound]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_src2(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_src2(size, size);
    gpu::Image<Pixel16scC4> gpu_dst(size, size);
    Pixel64fC4 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(cpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(gpu_src2, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.DivRoundC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.DivRoundC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC2> op(seed + 1);
    Pixel32sC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.DivRoundC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC3> op(seed + 1);
    Pixel32sC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.DivRoundC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_dst(size, size);
    Pixel64fC1 meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC1> op(seed + 1);
    Pixel16scC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.DivRoundC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_dst(size, size);
    cpu::Image<Pixel16scC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_dst(size, size);
    Pixel64fC2 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC2> op(seed + 1);
    Pixel16scC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.DivRoundC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_dst(size, size);
    Pixel64fC3 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC3> op(seed + 1);
    Pixel16scC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.DivRoundC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_dst(size, size);
    Pixel64fC4 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC4> op(seed + 1);
    Pixel16scC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.DivRoundDevC]")
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

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.DivRoundDevC]")
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
    mpp::cuda::DevVar<Pixel32sC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.DivRoundDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC2> op(seed + 1);
    Pixel32sC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32sC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.DivRoundDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC3> op(seed + 1);
    Pixel32sC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32sC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.DivRoundDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_dst(size, size);
    Pixel64fC1 meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC1> op(seed + 1);
    Pixel16scC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanError);
    CHECK(meanError < (10.0 / size / size)); // allow ten pixels to differ by 1
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.DivRoundDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_dst(size, size);
    cpu::Image<Pixel16scC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_dst(size, size);
    Pixel64fC2 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC2> op(seed + 1);
    Pixel16scC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.DivRoundDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_dst(size, size);
    Pixel64fC3 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC3> op(seed + 1);
    Pixel16scC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.DivRoundDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_dst(size, size);
    Pixel64fC4 meanErrors;
    double meanError;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC4> op(seed + 1);
    Pixel16scC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1.Div(constVal, cpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, gpu_dst, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_dst;
    cpu_dst.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // Inplace
    cpu_src1 >> cpu_dst;
    gpu_src1 >> gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.Div(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.Div(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.Div(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    // InplaceInv
    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesToEven);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesToEven);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::NearestTiesAwayFromZero);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardNegativeInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, 0, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));

    cpu_src1 << cpu_dst;
    gpu_src1 << gpu_dst;
    cpu_src1.DivInv(constVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_src1.DivInv(constDevVal, -2, RoundingMode::TowardPositiveInfinity);
    gpu_res << gpu_src1;
    cpu_src1.AverageError(gpu_res, meanErrors, meanError);
    CHECK(meanError < (10.0 / size / size));
}