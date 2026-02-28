#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/border.h>
#include <common/image/pitchException.h>
#include <common/image/pixelTypes.h>
#include <common/image/roiException.h>
#include <common/scratchBufferException.h>
#include <filesystem>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace gpu = mpp::image::cuda;

constexpr int size = 256;

TEST_CASE("8uC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Min.BufferException]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize() / 2);

    CHECK_THROWS_AS(gpu_src1.Min(gpu_dstMin, gpu_buffer), ScratchBufferException);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Min.ReducedRoi]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Min.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8uC1> cpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src1(sizeUneven, sizeUneven);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);

    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Min.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    gpu::ImageView<Pixel8uC1> gpu_src1Null(nullptr, {{size, size}, gpu_src1.Pitch()});
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);

    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    mpp::cuda::DevVarView<byte> gpu_bufferNull(nullptr, 0);

    CHECK_THROWS_AS(gpu_src1Null.Min(gpu_dstMin, gpu_buffer), NullPtrException);
    CHECK_THROWS_AS(gpu_src1.Min(gpu_dstMin, gpu_bufferNull), NullPtrException);
}

TEST_CASE("8uC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel8uC2 cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC2 gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8uC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel8uC3 cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC3 gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8uC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel8uC4 cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC4 gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8uC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel8uC4A cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC4A gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));

    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8uC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel8uC2 cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC2 gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8uC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel8uC3 cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC3 gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8uC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel8uC4 cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC4 gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8uC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel8uC4A cpu_dstMin;
    byte cpu_dstMinScalar;
    Pixel8uC4A gpu_resMin;
    byte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    Pixel8sC1 cpu_dstMin;
    Pixel8sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8sC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel8sC2 cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC2 gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel8sC3 cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC3 gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel8sC4 cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC4 gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel8sC4A cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC4A gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;
    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    Pixel8sC1 cpu_dstMin;
    Pixel8sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8sC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));

    Pixel8sC1 cpu_dstMin;
    Pixel8sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("8sC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel8sC2 cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC2 gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel8sC3 cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC3 gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel8sC4 cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC4 gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("8sC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel8sC4A cpu_dstMin;
    sbyte cpu_dstMinScalar;
    Pixel8sC4A gpu_resMin;
    sbyte gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16uC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel16uC2 cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC2 gpu_resMin;
    ushort gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel16uC3 cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC3 gpu_resMin;
    ushort gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch ::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel16uC4 cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC4 gpu_resMin;
    ushort gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel16uC4A cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC4A gpu_resMin;
    ushort gpu_resMinScalar;

    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16uC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);

    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));

    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16uC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel16uC2 cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC2 gpu_resMin;
    ushort gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel16uC3 cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC3 gpu_resMin;
    ushort gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel16uC4 cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC4 gpu_resMin;
    ushort gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16uC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel16uC4A cpu_dstMin;
    ushort cpu_dstMinScalar;
    Pixel16uC4A gpu_resMin;
    ushort gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    Pixel16sC1 cpu_dstMin;
    Pixel16sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16sC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel16sC2 cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC2 gpu_resMin;
    short gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel16sC3 cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC3 gpu_resMin;
    short gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel16sC4 cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC4 gpu_resMin;
    short gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel16sC4A cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC4A gpu_resMin;
    short gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    gpu_dstMinScalar >> gpu_resMinScalar;
    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    Pixel16sC1 cpu_dstMin;
    Pixel16sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16sC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);

    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));

    Pixel16sC1 cpu_dstMin;
    Pixel16sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16sC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel16sC2 cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC2 gpu_resMin;
    short gpu_resMinScalar;

    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel16sC3 cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC3 gpu_resMin;
    short gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel16sC4 cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC4 gpu_resMin;
    short gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16sC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel16sC4A cpu_dstMin;
    short cpu_dstMinScalar;
    Pixel16sC4A gpu_resMin;
    short gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    Pixel32uC1 cpu_dstMin;
    Pixel32uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32uC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel32uC2 cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC2 gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel32uC3 cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC3 gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel32uC4 cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC4 gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel32uC4A cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC4A gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;
    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);

    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    Pixel32uC1 cpu_dstMin;
    Pixel32uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32uC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel32uC1 cpu_dstMin;
    Pixel32uC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32uC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel32uC2 cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC2 gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel32uC3 cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC3 gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel32uC4 cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC4 gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32uC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel32uC4A cpu_dstMin;
    uint cpu_dstMinScalar;
    Pixel32uC4A gpu_resMin;
    uint gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    Pixel32sC1 cpu_dstMin;
    Pixel32sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32sC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel32sC2 cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC2 gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel32sC3 cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC3 gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);
    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel32sC4 cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC4 gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel32sC4A cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC4A gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    Pixel32sC1 cpu_dstMin;
    Pixel32sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32sC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel32sC1 cpu_dstMin;
    Pixel32sC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32sC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel32sC2 cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC2 gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel32sC3 cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC3 gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel32sC4 cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC4 gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32sC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel32sC4A cpu_dstMin;
    int cpu_dstMinScalar;
    Pixel32sC4A gpu_resMin;
    int gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    Pixel16fC1 cpu_dstMin;
    Pixel16fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16fC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel16fC2 cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC2 gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel16fC3 cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC3 gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel16fC4 cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC4 gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel16fC4A cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC4A gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    Pixel16fC1 cpu_dstMin;
    Pixel16fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16fC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel16fC1 cpu_dstMin;
    Pixel16fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16fC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel16fC2 cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC2 gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel16fC3 cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC3 gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel16fC4 cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC4 gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16fC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel16fC4A cpu_dstMin;
    HalfFp16 cpu_dstMinScalar;
    Pixel16fC4A gpu_resMin;
    HalfFp16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    Pixel16bfC1 cpu_dstMin;
    Pixel16bfC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16bfC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel16bfC2 cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC2 gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel16bfC3 cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC3 gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel16bfC4 cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC4 gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel16bfC4A cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC4A gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    Pixel16bfC1 cpu_dstMin;
    Pixel16bfC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);

    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));

    Pixel16bfC1 cpu_dstMin;
    Pixel16bfC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("16bfC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel16bfC2 cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC2 gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel16bfC3 cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC3 gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel16bfC4 cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC4 gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel16bfC4A cpu_dstMin;
    BFloat16 cpu_dstMinScalar;
    Pixel16bfC4A gpu_resMin;
    BFloat16 gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32fC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel32fC2 cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC2 gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel32fC3 cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC3 gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel32fC4 cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC4 gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel32fC4A cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC4A gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32fC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("32fC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel32fC2 cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC2 gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel32fC3 cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC3 gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel32fC4 cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC4 gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("32fC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel32fC4A cpu_dstMin;
    float cpu_dstMinScalar;
    Pixel32fC4A gpu_resMin;
    float gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC1", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dstMin;
    Pixel64fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.Min(cpu_dstMin);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("64fC2", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC2 gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC3", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC3 gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC4", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC4 gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC4A", "[CUDA.Statistics.Min]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC4A gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Min(gpu_dstMin, gpu_dstMinScalar, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.Min(cpu_dstMin, cpu_dstMinScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC1", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dstMin;
    Pixel64fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("64fC1", "[CUDA.Statistics.MinMasked.ReducedRoi]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);

    cpu_mask.SetRoi(Border(-1));
    gpu_mask.SetRoi(Border(-1));
    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));

    Pixel64fC1 cpu_dstMin;
    Pixel64fC1 gpu_resMin;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;

    cpu_src1.MinMasked(cpu_dstMin, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
}

TEST_CASE("64fC2", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC2 gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC3", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC3 gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC4", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC4 gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}

TEST_CASE("64fC4A", "[CUDA.Statistics.MinMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dstMin;
    double cpu_dstMinScalar;
    Pixel64fC4A gpu_resMin;
    double gpu_resMinScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MinMasked(gpu_dstMin, gpu_dstMinScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMinScalar >> gpu_resMinScalar;

    cpu_src1.MinMasked(cpu_dstMin, cpu_dstMinScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
}
