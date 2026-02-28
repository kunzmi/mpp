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
#include <common/half_fp16.h>
#include <common/image/pitchException.h>
#include <common/image/roiException.h>
#include <filesystem>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace gpu = mpp::image::cuda;

constexpr int size = 256;

TEST_CASE("8uC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel8uC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel8uC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel8sC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel8sC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel8sC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    cpu::Image<Pixel8sC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel8sC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel8sC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel8sC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    cpu::Image<Pixel8sC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel16uC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel16uC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel16sC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    cpu::Image<Pixel16sC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel16sC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    cpu::Image<Pixel16sC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32uC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel32uC1> cpu_src2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel32uC2> cpu_src2(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src2(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel32uC3> cpu_src2(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src2(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel32uC4> cpu_src2(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src2(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32uC4A> cpu_src2(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src2(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel32uC1> cpu_src2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel32uC2> cpu_src2(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src2(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel32uC3> cpu_src2(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src2(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel32uC4> cpu_src2(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src2(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32uC4A> cpu_src2(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src2(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_src2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_src2(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src2(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_src2(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src2(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_src2(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src2(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_src2(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src2(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_src2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_src2(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src2(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_src2(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src2(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_src2(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src2(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_src2(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src2(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    cpu::Image<Pixel16fC4A> cpu_src2(size, size);
    cpu::Image<Pixel16fC4A> cpu_dst(size, size);
    cpu::Image<Pixel16fC4A> gpu_res(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src2(size, size);
    gpu::Image<Pixel16fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_src2(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src2(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_src2(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src2(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_src2(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src2(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_src2(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src2(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    cpu::Image<Pixel16fC4A> cpu_src2(size, size);
    cpu::Image<Pixel16fC4A> cpu_dst(size, size);
    cpu::Image<Pixel16fC4A> gpu_res(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src2(size, size);
    gpu::Image<Pixel16fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16bfC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    cpu::Image<Pixel16bfC1> cpu_src2(size, size);
    cpu::Image<Pixel16bfC1> cpu_dst(size, size);
    cpu::Image<Pixel16bfC1> gpu_res(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src2(size, size);
    gpu::Image<Pixel16bfC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    cpu::Image<Pixel16bfC2> cpu_src2(size, size);
    cpu::Image<Pixel16bfC2> cpu_dst(size, size);
    cpu::Image<Pixel16bfC2> gpu_res(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src2(size, size);
    gpu::Image<Pixel16bfC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    cpu::Image<Pixel16bfC3> cpu_src2(size, size);
    cpu::Image<Pixel16bfC3> cpu_dst(size, size);
    cpu::Image<Pixel16bfC3> gpu_res(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src2(size, size);
    gpu::Image<Pixel16bfC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4> cpu_src2(size, size);
    cpu::Image<Pixel16bfC4> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4> gpu_res(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src2(size, size);
    gpu::Image<Pixel16bfC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4A> cpu_src2(size, size);
    cpu::Image<Pixel16bfC4A> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4A> gpu_res(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src2(size, size);
    gpu::Image<Pixel16bfC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    cpu::Image<Pixel16bfC1> cpu_src2(size, size);
    cpu::Image<Pixel16bfC1> cpu_dst(size, size);
    cpu::Image<Pixel16bfC1> gpu_res(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src2(size, size);
    gpu::Image<Pixel16bfC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    cpu::Image<Pixel16bfC2> cpu_src2(size, size);
    cpu::Image<Pixel16bfC2> cpu_dst(size, size);
    cpu::Image<Pixel16bfC2> gpu_res(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src2(size, size);
    gpu::Image<Pixel16bfC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    cpu::Image<Pixel16bfC3> cpu_src2(size, size);
    cpu::Image<Pixel16bfC3> cpu_dst(size, size);
    cpu::Image<Pixel16bfC3> gpu_res(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src2(size, size);
    gpu::Image<Pixel16bfC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4> cpu_src2(size, size);
    cpu::Image<Pixel16bfC4> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4> gpu_res(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src2(size, size);
    gpu::Image<Pixel16bfC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4A> cpu_src2(size, size);
    cpu::Image<Pixel16bfC4A> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4A> gpu_res(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src2(size, size);
    gpu::Image<Pixel16bfC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AddWeighted]")
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
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("64fC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_src2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_src2(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src2(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_src2(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src2(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_src2(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src2(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4A", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_src2(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src2(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_src2(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src2(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_src2(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src2(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_src2(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src2(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_src2(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src2(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4A", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_src2(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src2(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_src2(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_src2(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_src2(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_src2(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_src2(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_src2(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_src2(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_src2(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_src2(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_src2(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_src2(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_src2(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_src2(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_src2(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_src2(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_src2(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC1> cpu_src1(size, size);
    cpu::Image<Pixel32scC1> cpu_src2(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32scC1> gpu_src1(size, size);
    gpu::Image<Pixel32scC1> gpu_src2(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC2> cpu_src1(size, size);
    cpu::Image<Pixel32scC2> cpu_src2(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32scC2> gpu_src1(size, size);
    gpu::Image<Pixel32scC2> gpu_src2(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC3> cpu_src1(size, size);
    cpu::Image<Pixel32scC3> cpu_src2(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32scC3> gpu_src1(size, size);
    gpu::Image<Pixel32scC3> gpu_src2(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC4> cpu_src1(size, size);
    cpu::Image<Pixel32scC4> cpu_src2(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32scC4> gpu_src1(size, size);
    gpu::Image<Pixel32scC4> gpu_src2(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC1> cpu_src1(size, size);
    cpu::Image<Pixel32scC1> cpu_src2(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32scC1> gpu_src1(size, size);
    gpu::Image<Pixel32scC1> gpu_src2(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC2> cpu_src1(size, size);
    cpu::Image<Pixel32scC2> cpu_src2(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32scC2> gpu_src1(size, size);
    gpu::Image<Pixel32scC2> gpu_src2(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC3> cpu_src1(size, size);
    cpu::Image<Pixel32scC3> cpu_src2(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32scC3> gpu_src1(size, size);
    gpu::Image<Pixel32scC3> gpu_src2(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC4> cpu_src1(size, size);
    cpu::Image<Pixel32scC4> cpu_src2(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32scC4> gpu_src1(size, size);
    gpu::Image<Pixel32scC4> gpu_src2(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC1", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_src2(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32fcC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_src2(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC2", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_src2(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32fcC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_src2(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC3", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_src2(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32fcC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_src2(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC4", "[CUDA.Arithmetic.AddWeighted]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_src2(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32fcC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_src2(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_src2, cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_src2, gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC1", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_src2(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32fcC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_src2(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC2", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_src2(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32fcC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_src2(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC3", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_src2(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32fcC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_src2(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC4", "[CUDA.Arithmetic.AddWeightedMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_src2(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32fcC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_src2(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_src2, cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_src2, gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("8sC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16sC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32uC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32uC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00000001));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    cpu::Image<Pixel16fC4A> cpu_dst(size, size);
    cpu::Image<Pixel16fC4A> gpu_res(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_hf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_hf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16fC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    cpu::Image<Pixel16fC4A> cpu_dst(size, size);
    cpu::Image<Pixel16fC4A> gpu_res(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_hf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_hf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_hf));
}

TEST_CASE("16bfC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    cpu::Image<Pixel16bfC1> cpu_dst(size, size);
    cpu::Image<Pixel16bfC1> gpu_res(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    cpu::Image<Pixel16bfC2> cpu_dst(size, size);
    cpu::Image<Pixel16bfC2> gpu_res(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    cpu::Image<Pixel16bfC3> cpu_dst(size, size);
    cpu::Image<Pixel16bfC3> gpu_res(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4> gpu_res(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4A> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4A> gpu_res(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3_bf);
    gpu_src1.AddWeighted(gpu_dst, 0.3_bf);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    cpu::Image<Pixel16bfC1> cpu_dst(size, size);
    cpu::Image<Pixel16bfC1> gpu_res(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    cpu::Image<Pixel16bfC2> cpu_dst(size, size);
    cpu::Image<Pixel16bfC2> gpu_res(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    cpu::Image<Pixel16bfC3> cpu_dst(size, size);
    cpu::Image<Pixel16bfC3> gpu_res(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4> gpu_res(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("16bfC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    cpu::Image<Pixel16bfC4A> cpu_dst(size, size);
    cpu::Image<Pixel16bfC4A> gpu_res(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3_bf, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3_bf, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.001_bf));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("64fC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4A", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_dst(size, size);
    cpu::Image<Pixel64fC1> gpu_res(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_dst(size, size);
    cpu::Image<Pixel64fC2> gpu_res(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_dst(size, size);
    cpu::Image<Pixel64fC3> gpu_res(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_dst(size, size);
    cpu::Image<Pixel64fC4> gpu_res(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("64fC4A", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    cpu::Image<Pixel64fC4A> cpu_dst(size, size);
    cpu::Image<Pixel64fC4A> gpu_res(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.000000001));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32scC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32scC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32scC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32scC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32scC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32scC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32scC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32scC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32scC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32scC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32scC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC1", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32fcC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC2", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32fcC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC3", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32fcC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC4", "[CUDA.Arithmetic.AddWeightedInplace]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32fcC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    gpu_src1.AddWeighted(gpu_dst, 0.3f);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC1", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32fcC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC2", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32fcC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC3", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32fcC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}

TEST_CASE("32fcC4", "[CUDA.Arithmetic.AddWeightedInplaceMasked]")
{
    const uint seed = Catch::getSeed();

    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    cpu::Image<Pixel32fcC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32fcC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_mask(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 2);

    cpu_mask >> gpu_mask;
    cpu_src1 >> gpu_src1;
    cpu_dst >> gpu_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    gpu_src1.AddWeightedMasked(gpu_dst, 0.3f, gpu_mask);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.00001f));
}