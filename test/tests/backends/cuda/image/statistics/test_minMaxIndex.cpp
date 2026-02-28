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

TEST_CASE("8uC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8uC1 gpu_resMin;
    Pixel8uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMaxIndex.BufferException]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize() / 2);

    CHECK_THROWS_AS(gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer), ScratchBufferException);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMaxIndex.ReducedRoi]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);

    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8uC1 gpu_resMin;
    Pixel8uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMaxIndex.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8uC1> cpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src1(sizeUneven, sizeUneven);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8uC1 gpu_resMin;
    Pixel8uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMaxIndex.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    gpu::ImageView<Pixel8uC1> gpu_src1Null(nullptr, {{size, size}, gpu_src1.Pitch()});
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());
    mpp::cuda::DevVarView<byte> gpu_bufferNull(nullptr, 0);

    CHECK_THROWS_AS(gpu_src1Null.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer), NullPtrException);
    CHECK_THROWS_AS(gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_bufferNull), NullPtrException);
}

TEST_CASE("8uC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel8uC2 cpu_dstMin;
    Pixel8uC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC2 gpu_resMin;
    Pixel8uC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8uC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel8uC3 cpu_dstMin;
    Pixel8uC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC3 gpu_resMin;
    Pixel8uC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8uC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel8uC4 cpu_dstMin;
    Pixel8uC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC4 gpu_resMin;
    Pixel8uC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8uC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel8uC4A cpu_dstMin;
    Pixel8uC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC4A gpu_resMin;
    Pixel8uC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8uC1 gpu_resMin;
    Pixel8uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8uC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel8uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8uC1 gpu_resMin;
    Pixel8uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8uC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel8uC2 cpu_dstMin;
    Pixel8uC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC2 gpu_resMin;
    Pixel8uC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8uC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel8uC3 cpu_dstMin;
    Pixel8uC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC3 gpu_resMin;
    Pixel8uC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8uC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel8uC4 cpu_dstMin;
    Pixel8uC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC4 gpu_resMin;
    Pixel8uC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8uC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel8uC4A cpu_dstMin;
    Pixel8uC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    byte cpu_dstMinScalar;
    byte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8uC4A gpu_resMin;
    Pixel8uC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    byte gpu_resMinScalar;
    byte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<byte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<byte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    Pixel8sC1 cpu_dstMin;
    Pixel8sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8sC1 gpu_resMin;
    Pixel8sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8sC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel8sC2 cpu_dstMin;
    Pixel8sC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC2 gpu_resMin;
    Pixel8sC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel8sC3 cpu_dstMin;
    Pixel8sC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC3 gpu_resMin;
    Pixel8sC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel8sC4 cpu_dstMin;
    Pixel8sC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC4 gpu_resMin;
    Pixel8sC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel8sC4A cpu_dstMin;
    Pixel8sC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC4A gpu_resMin;
    Pixel8sC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);

    Pixel8sC1 cpu_dstMin;
    Pixel8sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8sC1 gpu_resMin;
    Pixel8sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8sC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel8sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel8sC1 gpu_resMin;
    Pixel8sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("8sC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel8sC2 cpu_dstMin;
    Pixel8sC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC2 gpu_resMin;
    Pixel8sC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel8sC3 cpu_dstMin;
    Pixel8sC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC3 gpu_resMin;
    Pixel8sC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel8sC4 cpu_dstMin;
    Pixel8sC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC4 gpu_resMin;
    Pixel8sC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("8sC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel8sC4A cpu_dstMin;
    Pixel8sC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    sbyte cpu_dstMinScalar;
    sbyte cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel8sC4A gpu_resMin;
    Pixel8sC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    sbyte gpu_resMinScalar;
    sbyte gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel8sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<sbyte> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<sbyte> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16uC1 gpu_resMin;
    Pixel16uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16uC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel16uC2 cpu_dstMin;
    Pixel16uC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC2 gpu_resMin;
    Pixel16uC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel16uC3 cpu_dstMin;
    Pixel16uC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC3 gpu_resMin;
    Pixel16uC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch ::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel16uC4 cpu_dstMin;
    Pixel16uC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC4 gpu_resMin;
    Pixel16uC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel16uC4A cpu_dstMin;
    Pixel16uC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC4A gpu_resMin;
    Pixel16uC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);

    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16uC1 gpu_resMin;
    Pixel16uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16uC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel16uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16uC1 gpu_resMin;
    Pixel16uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16uC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel16uC2 cpu_dstMin;
    Pixel16uC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC2 gpu_resMin;
    Pixel16uC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel16uC3 cpu_dstMin;
    Pixel16uC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC3 gpu_resMin;
    Pixel16uC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel16uC4 cpu_dstMin;
    Pixel16uC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC4 gpu_resMin;
    Pixel16uC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16uC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel16uC4A cpu_dstMin;
    Pixel16uC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    ushort cpu_dstMinScalar;
    ushort cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16uC4A gpu_resMin;
    Pixel16uC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    ushort gpu_resMinScalar;
    ushort gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<ushort> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<ushort> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    Pixel16sC1 cpu_dstMin;
    Pixel16sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16sC1 gpu_resMin;
    Pixel16sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16sC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel16sC2 cpu_dstMin;
    Pixel16sC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC2 gpu_resMin;
    Pixel16sC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel16sC3 cpu_dstMin;
    Pixel16sC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC3 gpu_resMin;
    Pixel16sC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel16sC4 cpu_dstMin;
    Pixel16sC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC4 gpu_resMin;
    Pixel16sC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel16sC4A cpu_dstMin;
    Pixel16sC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC4A gpu_resMin;
    Pixel16sC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);

    Pixel16sC1 cpu_dstMin;
    Pixel16sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16sC1 gpu_resMin;
    Pixel16sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16sC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel16sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16sC1 gpu_resMin;
    Pixel16sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16sC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel16sC2 cpu_dstMin;
    Pixel16sC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC2 gpu_resMin;
    Pixel16sC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel16sC3 cpu_dstMin;
    Pixel16sC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC3 gpu_resMin;
    Pixel16sC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel16sC4 cpu_dstMin;
    Pixel16sC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC4 gpu_resMin;
    Pixel16sC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16sC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel16sC4A cpu_dstMin;
    Pixel16sC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    short cpu_dstMinScalar;
    short cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16sC4A gpu_resMin;
    Pixel16sC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    short gpu_resMinScalar;
    short gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<short> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<short> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    Pixel32uC1 cpu_dstMin;
    Pixel32uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32uC1 gpu_resMin;
    Pixel32uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32uC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel32uC2 cpu_dstMin;
    Pixel32uC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC2 gpu_resMin;
    Pixel32uC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel32uC3 cpu_dstMin;
    Pixel32uC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC3 gpu_resMin;
    Pixel32uC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel32uC4 cpu_dstMin;
    Pixel32uC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC4 gpu_resMin;
    Pixel32uC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel32uC4A cpu_dstMin;
    Pixel32uC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC4A gpu_resMin;
    Pixel32uC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);

    Pixel32uC1 cpu_dstMin;
    Pixel32uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32uC1 gpu_resMin;
    Pixel32uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32uC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel32uC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32uC1 gpu_resMin;
    Pixel32uC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32uC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel32uC2 cpu_dstMin;
    Pixel32uC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC2 gpu_resMin;
    Pixel32uC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel32uC3 cpu_dstMin;
    Pixel32uC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC3 gpu_resMin;
    Pixel32uC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel32uC4 cpu_dstMin;
    Pixel32uC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC4 gpu_resMin;
    Pixel32uC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32uC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel32uC4A cpu_dstMin;
    Pixel32uC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    uint cpu_dstMinScalar;
    uint cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32uC4A gpu_resMin;
    Pixel32uC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    uint gpu_resMinScalar;
    uint gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32uC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<uint> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<uint> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    Pixel32sC1 cpu_dstMin;
    Pixel32sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32sC1 gpu_resMin;
    Pixel32sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32sC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel32sC2 cpu_dstMin;
    Pixel32sC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC2 gpu_resMin;
    Pixel32sC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel32sC3 cpu_dstMin;
    Pixel32sC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC3 gpu_resMin;
    Pixel32sC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel32sC4 cpu_dstMin;
    Pixel32sC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC4 gpu_resMin;
    Pixel32sC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel32sC4A cpu_dstMin;
    Pixel32sC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC4A gpu_resMin;
    Pixel32sC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);

    Pixel32sC1 cpu_dstMin;
    Pixel32sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32sC1 gpu_resMin;
    Pixel32sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32sC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel32sC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32sC1 gpu_resMin;
    Pixel32sC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32sC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel32sC2 cpu_dstMin;
    Pixel32sC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC2 gpu_resMin;
    Pixel32sC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel32sC3 cpu_dstMin;
    Pixel32sC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC3 gpu_resMin;
    Pixel32sC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel32sC4 cpu_dstMin;
    Pixel32sC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC4 gpu_resMin;
    Pixel32sC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32sC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel32sC4A cpu_dstMin;
    Pixel32sC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    int cpu_dstMinScalar;
    int cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32sC4A gpu_resMin;
    Pixel32sC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    int gpu_resMinScalar;
    int gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32sC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<int> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<int> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    Pixel16fC1 cpu_dstMin;
    Pixel16fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16fC1 gpu_resMin;
    Pixel16fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16fC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel16fC2 cpu_dstMin;
    Pixel16fC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC2 gpu_resMin;
    Pixel16fC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel16fC3 cpu_dstMin;
    Pixel16fC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC3 gpu_resMin;
    Pixel16fC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel16fC4 cpu_dstMin;
    Pixel16fC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC4 gpu_resMin;
    Pixel16fC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel16fC4A cpu_dstMin;
    Pixel16fC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC4A gpu_resMin;
    Pixel16fC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);

    Pixel16fC1 cpu_dstMin;
    Pixel16fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16fC1 gpu_resMin;
    Pixel16fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16fC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel16fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16fC1 gpu_resMin;
    Pixel16fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16fC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel16fC2 cpu_dstMin;
    Pixel16fC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC2 gpu_resMin;
    Pixel16fC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel16fC3 cpu_dstMin;
    Pixel16fC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC3 gpu_resMin;
    Pixel16fC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel16fC4 cpu_dstMin;
    Pixel16fC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC4 gpu_resMin;
    Pixel16fC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16fC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel16fC4A cpu_dstMin;
    Pixel16fC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    HalfFp16 cpu_dstMinScalar;
    HalfFp16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16fC4A gpu_resMin;
    Pixel16fC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    HalfFp16 gpu_resMinScalar;
    HalfFp16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<HalfFp16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    Pixel16bfC1 cpu_dstMin;
    Pixel16bfC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16bfC1 gpu_resMin;
    Pixel16bfC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16bfC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel16bfC2 cpu_dstMin;
    Pixel16bfC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC2 gpu_resMin;
    Pixel16bfC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel16bfC3 cpu_dstMin;
    Pixel16bfC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC3 gpu_resMin;
    Pixel16bfC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel16bfC4 cpu_dstMin;
    Pixel16bfC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC4 gpu_resMin;
    Pixel16bfC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel16bfC4A cpu_dstMin;
    Pixel16bfC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC4A gpu_resMin;
    Pixel16bfC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);

    Pixel16bfC1 cpu_dstMin;
    Pixel16bfC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16bfC1 gpu_resMin;
    Pixel16bfC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16bfC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel16bfC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel16bfC1 gpu_resMin;
    Pixel16bfC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("16bfC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel16bfC2 cpu_dstMin;
    Pixel16bfC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC2 gpu_resMin;
    Pixel16bfC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel16bfC3 cpu_dstMin;
    Pixel16bfC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC3 gpu_resMin;
    Pixel16bfC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel16bfC4 cpu_dstMin;
    Pixel16bfC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC4 gpu_resMin;
    Pixel16bfC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel16bfC4A cpu_dstMin;
    Pixel16bfC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    BFloat16 cpu_dstMinScalar;
    BFloat16 cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel16bfC4A gpu_resMin;
    Pixel16bfC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    BFloat16 gpu_resMinScalar;
    BFloat16 gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel16bfC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<BFloat16> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<BFloat16> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32fC1 gpu_resMin;
    Pixel32fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32fC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel32fC2 cpu_dstMin;
    Pixel32fC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC2 gpu_resMin;
    Pixel32fC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel32fC3 cpu_dstMin;
    Pixel32fC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC3 gpu_resMin;
    Pixel32fC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel32fC4 cpu_dstMin;
    Pixel32fC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC4 gpu_resMin;
    Pixel32fC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel32fC4A cpu_dstMin;
    Pixel32fC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC4A gpu_resMin;
    Pixel32fC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);

    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32fC1 gpu_resMin;
    Pixel32fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32fC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel32fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel32fC1 gpu_resMin;
    Pixel32fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("32fC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel32fC2 cpu_dstMin;
    Pixel32fC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC2 gpu_resMin;
    Pixel32fC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel32fC3 cpu_dstMin;
    Pixel32fC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC3 gpu_resMin;
    Pixel32fC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel32fC4 cpu_dstMin;
    Pixel32fC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC4 gpu_resMin;
    Pixel32fC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("32fC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel32fC4A cpu_dstMin;
    Pixel32fC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    float cpu_dstMinScalar;
    float cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel32fC4A gpu_resMin;
    Pixel32fC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    float gpu_resMinScalar;
    float gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<float> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<float> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC1", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dstMin;
    Pixel64fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel64fC1 gpu_resMin;
    Pixel64fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("64fC2", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dstMin;
    Pixel64fC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC2 gpu_resMin;
    Pixel64fC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC3", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dstMin;
    Pixel64fC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC3 gpu_resMin;
    Pixel64fC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC4", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dstMin;
    Pixel64fC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC4 gpu_resMin;
    Pixel64fC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC4A", "[CUDA.Statistics.MinMaxIndex]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dstMin;
    Pixel64fC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC4A gpu_resMin;
    Pixel64fC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndex(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar, gpu_dstIndexScalar,
                         gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar, cpu_dstIndexScalar);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC1", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);

    Pixel64fC1 cpu_dstMin;
    Pixel64fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel64fC1 gpu_resMin;
    Pixel64fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("64fC1", "[CUDA.Statistics.MinMaxIndexMasked.ReducedRoi]")
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
    Pixel64fC1 cpu_dstMax;
    IndexMinMax cpu_dstIndex;
    Pixel64fC1 gpu_resMin;
    Pixel64fC1 gpu_resMax;
    IndexMinMax gpu_resIndex;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC1> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex.IndexMin == gpu_resIndex.IndexMin);
    CHECK(cpu_dstIndex.IndexMax == gpu_resIndex.IndexMax);
}

TEST_CASE("64fC2", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dstMin;
    Pixel64fC2 cpu_dstMax;
    IndexMinMax cpu_dstIndex[2];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC2 gpu_resMin;
    Pixel64fC2 gpu_resMax;
    IndexMinMax gpu_resIndex[2];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC2> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(2);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC3", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dstMin;
    Pixel64fC3 cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC3 gpu_resMin;
    Pixel64fC3 gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC3> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC4", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dstMin;
    Pixel64fC4 cpu_dstMax;
    IndexMinMax cpu_dstIndex[4];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC4 gpu_resMin;
    Pixel64fC4 gpu_resMax;
    IndexMinMax gpu_resIndex[4];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC4> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(4);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstIndex[3].IndexMin == gpu_resIndex[3].IndexMin);
    CHECK(cpu_dstIndex[3].IndexMax == gpu_resIndex[3].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}

TEST_CASE("64fC4A", "[CUDA.Statistics.MinMaxIndexMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dstMin;
    Pixel64fC4A cpu_dstMax;
    IndexMinMax cpu_dstIndex[3];
    double cpu_dstMinScalar;
    double cpu_dstMaxScalar;
    IndexMinMaxChannel cpu_dstIndexScalar;
    Pixel64fC4A gpu_resMin;
    Pixel64fC4A gpu_resMax;
    IndexMinMax gpu_resIndex[3];
    double gpu_resMinScalar;
    double gpu_resMaxScalar;
    IndexMinMaxChannel gpu_resIndexScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMin(1);
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dstMax(1);
    mpp::cuda::DevVar<IndexMinMax> gpu_dstIndex(3);
    mpp::cuda::DevVar<double> gpu_dstMinScalar(1);
    mpp::cuda::DevVar<double> gpu_dstMaxScalar(1);
    mpp::cuda::DevVar<IndexMinMaxChannel> gpu_dstIndexScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.MinMaxIndexBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.MinMaxIndexMasked(gpu_dstMin, gpu_dstMax, gpu_dstIndex, gpu_dstMinScalar, gpu_dstMaxScalar,
                               gpu_dstIndexScalar, gpu_mask, gpu_buffer);
    gpu_dstMin >> gpu_resMin;
    gpu_dstMax >> gpu_resMax;
    gpu_dstIndex >> gpu_resIndex;
    gpu_dstMinScalar >> gpu_resMinScalar;
    gpu_dstMaxScalar >> gpu_resMaxScalar;
    gpu_dstIndexScalar >> gpu_resIndexScalar;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_dstIndex, cpu_dstMinScalar, cpu_dstMaxScalar,
                               cpu_dstIndexScalar, cpu_mask);

    CHECK(cpu_dstMin == gpu_resMin);
    CHECK(cpu_dstMax == gpu_resMax);
    CHECK(cpu_dstIndex[0].IndexMin == gpu_resIndex[0].IndexMin);
    CHECK(cpu_dstIndex[0].IndexMax == gpu_resIndex[0].IndexMax);
    CHECK(cpu_dstIndex[1].IndexMin == gpu_resIndex[1].IndexMin);
    CHECK(cpu_dstIndex[1].IndexMax == gpu_resIndex[1].IndexMax);
    CHECK(cpu_dstIndex[2].IndexMin == gpu_resIndex[2].IndexMin);
    CHECK(cpu_dstIndex[2].IndexMax == gpu_resIndex[2].IndexMax);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstMaxScalar == gpu_resMaxScalar);
    CHECK(cpu_dstMinScalar == gpu_resMinScalar);
    CHECK(cpu_dstIndexScalar.IndexMin == gpu_resIndexScalar.IndexMin);
    CHECK(cpu_dstIndexScalar.IndexMax == gpu_resIndexScalar.IndexMax);
    CHECK(cpu_dstIndexScalar.ChannelMin == gpu_resIndexScalar.ChannelMin);
    CHECK(cpu_dstIndexScalar.ChannelMax == gpu_resIndexScalar.ChannelMax);
}
