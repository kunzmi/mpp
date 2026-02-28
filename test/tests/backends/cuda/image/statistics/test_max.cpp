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

TEST_CASE("8uC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    Pixel8uC1 cpu_dstMax;
    Pixel8uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Max.BufferException]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize() / 2);

    CHECK_THROWS_AS(gpu_src1.Max(gpu_dstMax, gpu_buffer), ScratchBufferException);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Max.ReducedRoi]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel8uC1 cpu_dstMax;
    Pixel8uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Max.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8uC1> cpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src1(sizeUneven, sizeUneven);
    Pixel8uC1 cpu_dstMax;
    Pixel8uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);

    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.Max.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    gpu::ImageView<Pixel8uC1> gpu_src1Null(nullptr, {{size, size}, gpu_src1.Pitch()});
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);

    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    mpp::cuda::DevVarView<byte> gpu_bufferNull(nullptr, 0);

    CHECK_THROWS_AS(gpu_src1Null.Max(gpu_dstMax, gpu_buffer), NullPtrException);
    CHECK_THROWS_AS(gpu_src1.Max(gpu_dstMax, gpu_bufferNull), NullPtrException);
}

TEST_CASE("8uC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel8uC2 cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC2 gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8uC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel8uC3 cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC3 gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8uC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel8uC4 cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC4 gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8uC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel8uC4A cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC4A gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    Pixel8uC1 cpu_dstMax;
    Pixel8uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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

    Pixel8uC1 cpu_dstMax;
    Pixel8uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8uC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel8uC2 cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC2 gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8uC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel8uC3 cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC3 gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8uC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel8uC4 cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC4 gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8uC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel8uC4A cpu_dstMax;
    byte cpu_dstMaxScalar;
    Pixel8uC4A gpu_resMax;
    byte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    Pixel8sC1 cpu_dstMax;
    Pixel8sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8sC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel8sC2 cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC2 gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel8sC3 cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC3 gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel8sC4 cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC4 gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel8sC4A cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC4A gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;
    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    Pixel8sC1 cpu_dstMax;
    Pixel8sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8sC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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

    Pixel8sC1 cpu_dstMax;
    Pixel8sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("8sC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel8sC2 cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC2 gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel8sC3 cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC3 gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel8sC4 cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC4 gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("8sC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel8sC4A cpu_dstMax;
    sbyte cpu_dstMaxScalar;
    Pixel8sC4A gpu_resMax;
    sbyte gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    Pixel16uC1 cpu_dstMax;
    Pixel16uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16uC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel16uC2 cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC2 gpu_resMax;
    ushort gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel16uC3 cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC3 gpu_resMax;
    ushort gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch ::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel16uC4 cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC4 gpu_resMax;
    ushort gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel16uC4A cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC4A gpu_resMax;
    ushort gpu_resMaxScalar;

    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    Pixel16uC1 cpu_dstMax;
    Pixel16uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16uC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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

    Pixel16uC1 cpu_dstMax;
    Pixel16uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16uC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel16uC2 cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC2 gpu_resMax;
    ushort gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel16uC3 cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC3 gpu_resMax;
    ushort gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel16uC4 cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC4 gpu_resMax;
    ushort gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16uC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel16uC4A cpu_dstMax;
    ushort cpu_dstMaxScalar;
    Pixel16uC4A gpu_resMax;
    ushort gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    Pixel16sC1 cpu_dstMax;
    Pixel16sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16sC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel16sC2 cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC2 gpu_resMax;
    short gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel16sC3 cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC3 gpu_resMax;
    short gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel16sC4 cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC4 gpu_resMax;
    short gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel16sC4A cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC4A gpu_resMax;
    short gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    gpu_dstMaxScalar >> gpu_resMaxScalar;
    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    Pixel16sC1 cpu_dstMax;
    Pixel16sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16sC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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

    Pixel16sC1 cpu_dstMax;
    Pixel16sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16sC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel16sC2 cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC2 gpu_resMax;
    short gpu_resMaxScalar;

    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel16sC3 cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC3 gpu_resMax;
    short gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel16sC4 cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC4 gpu_resMax;
    short gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16sC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel16sC4A cpu_dstMax;
    short cpu_dstMaxScalar;
    Pixel16sC4A gpu_resMax;
    short gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    Pixel32uC1 cpu_dstMax;
    Pixel32uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32uC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel32uC2 cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC2 gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel32uC3 cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC3 gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel32uC4 cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC4 gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel32uC4A cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC4A gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;
    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);

    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    Pixel32uC1 cpu_dstMax;
    Pixel32uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32uC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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
    Pixel32uC1 cpu_dstMax;
    Pixel32uC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32uC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel32uC2 cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC2 gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel32uC3 cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC3 gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel32uC4 cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC4 gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32uC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel32uC4A cpu_dstMax;
    uint cpu_dstMaxScalar;
    Pixel32uC4A gpu_resMax;
    uint gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    Pixel32sC1 cpu_dstMax;
    Pixel32sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32sC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel32sC2 cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC2 gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel32sC3 cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC3 gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel32sC4 cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC4 gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel32sC4A cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC4A gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    Pixel32sC1 cpu_dstMax;
    Pixel32sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32sC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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
    Pixel32sC1 cpu_dstMax;
    Pixel32sC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32sC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel32sC2 cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC2 gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel32sC3 cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC3 gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel32sC4 cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC4 gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32sC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel32sC4A cpu_dstMax;
    int cpu_dstMaxScalar;
    Pixel32sC4A gpu_resMax;
    int gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    Pixel16fC1 cpu_dstMax;
    Pixel16fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16fC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel16fC2 cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC2 gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel16fC3 cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC3 gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel16fC4 cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC4 gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel16fC4A cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC4A gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    Pixel16fC1 cpu_dstMax;
    Pixel16fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16fC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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
    Pixel16fC1 cpu_dstMax;
    Pixel16fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16fC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel16fC2 cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC2 gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel16fC3 cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC3 gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel16fC4 cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC4 gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16fC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel16fC4A cpu_dstMax;
    HalfFp16 cpu_dstMaxScalar;
    Pixel16fC4A gpu_resMax;
    HalfFp16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    Pixel16bfC1 cpu_dstMax;
    Pixel16bfC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16bfC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel16bfC2 cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC2 gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel16bfC3 cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC3 gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel16bfC4 cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC4 gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel16bfC4A cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC4A gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    Pixel16bfC1 cpu_dstMax;
    Pixel16bfC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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

    Pixel16bfC1 cpu_dstMax;
    Pixel16bfC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("16bfC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel16bfC2 cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC2 gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel16bfC3 cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC3 gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel16bfC4 cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC4 gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel16bfC4A cpu_dstMax;
    BFloat16 cpu_dstMaxScalar;
    Pixel16bfC4A gpu_resMax;
    BFloat16 gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    Pixel32fC1 cpu_dstMax;
    Pixel32fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32fC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel32fC2 cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC2 gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel32fC3 cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC3 gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel32fC4 cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC4 gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel32fC4A cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC4A gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    Pixel32fC1 cpu_dstMax;
    Pixel32fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32fC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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
    Pixel32fC1 cpu_dstMax;
    Pixel32fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());
    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("32fC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel32fC2 cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC2 gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel32fC3 cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC3 gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel32fC4 cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC4 gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("32fC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel32fC4A cpu_dstMax;
    float cpu_dstMaxScalar;
    Pixel32fC4A gpu_resMax;
    float gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC1", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dstMax;
    Pixel64fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.Max(cpu_dstMax);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("64fC2", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC2 gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC3", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC3 gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC4", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC4 gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC4A", "[CUDA.Statistics.Max]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC4A gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.Max(gpu_dstMax, gpu_dstMaxScalar, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.Max(cpu_dstMax, cpu_dstMaxScalar);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC1", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dstMax;
    Pixel64fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("64fC1", "[CUDA.Statistics.MaxMasked.ReducedRoi]")
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

    Pixel64fC1 cpu_dstMax;
    Pixel64fC1 gpu_resMax;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
}

TEST_CASE("64fC2", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC2 gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC3", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC3 gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC4", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC4 gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}

TEST_CASE("64fC4A", "[CUDA.Statistics.MaxMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dstMax;
    double cpu_dstMaxScalar;
    Pixel64fC4A gpu_resMax;
    double gpu_resMaxScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MaxBufferSize());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;

    gpu_src1.MaxMasked(gpu_dstMax, gpu_dstMaxScalar, gpu_mask, gpu_buffer);
    gpu_dstMax >> gpu_resMax;
    gpu_dstMaxScalar >> gpu_resMaxScalar;

    cpu_src1.MaxMasked(cpu_dstMax, cpu_dstMaxScalar, cpu_mask);

    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
}
