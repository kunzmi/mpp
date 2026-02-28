#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
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

TEST_CASE("8uC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("8uC1", "[CUDA.Statistics.NormInf.BufferException]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize() / 2);

    CHECK_THROWS_AS(gpu_src1.NormInf(gpu_dst, gpu_buffer), ScratchBufferException);
}

TEST_CASE("8uC1", "[CUDA.Statistics.NormInf.ReducedRoi]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);

    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    cpu_src1.SetRoi(Border(-1));
    gpu_src1.SetRoi(Border(-1));
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("8uC1", "[CUDA.Statistics.NormInf.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8uC1> cpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src1(sizeUneven, sizeUneven);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("8uC1", "[CUDA.Statistics.NormInf.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    gpu::ImageView<Pixel8uC1> gpu_src1Null(nullptr, {{size, size}, gpu_src1.Pitch()});
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());
    mpp::cuda::DevVarView<byte> gpu_bufferNull(nullptr, 0);

    CHECK_THROWS_AS(gpu_src1Null.NormInf(gpu_dst, gpu_buffer), NullPtrException);
    CHECK_THROWS_AS(gpu_src1.NormInf(gpu_dst, gpu_bufferNull), NullPtrException);
}

TEST_CASE("8uC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8uC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8uC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8uC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8uC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC1> cpu_src1(size, size);

    gpu::Image<Pixel8uC1> gpu_src1(size, size);

    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("8uC1", "[CUDA.Statistics.NormInfMasked.ReducedRoi]")
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

    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("8uC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8uC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8uC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8uC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("8sC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("8sC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("8sC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16uC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16uC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16uC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16sC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16sC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16sC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(100); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("32uC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("32uC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32uC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("32sC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("32sC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32sC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1.Div(10000); // reduce value range a bit to avoid outliers

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel32fC1 gpu_res;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16fC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel32fC2 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel32fC3 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel32fC4 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel32fC4A gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel32fC1 gpu_res;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16fC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel32fC2 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel32fC3 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel32fC4 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16fC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16fC4A> cpu_src1(size, size);
    gpu::Image<Pixel16fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel32fC4A gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel32fC1 gpu_res;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16bfC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel32fC2 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel32fC3 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel32fC4 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel32fC4A gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel32fC1 gpu_res;
    mpp::cuda::DevVar<Pixel32fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("16bfC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC2> cpu_src1(size, size);
    gpu::Image<Pixel16bfC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel32fC2 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC2> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC3> cpu_src1(size, size);
    gpu::Image<Pixel16bfC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel32fC3 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC3> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel32fC4 gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("16bfC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel16bfC4A> cpu_src1(size, size);
    gpu::Image<Pixel16bfC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel32fC4A gpu_res;
    double cpu_dstScalar;
    float gpu_resScalar;
    mpp::cuda::DevVar<Pixel32fC4A> gpu_dst(1);
    mpp::cuda::DevVar<float> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("32fC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("32fC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("32fC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC1", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInf(cpu_dst);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("64fC2", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC3", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC4", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC4A", "[CUDA.Statistics.NormInf]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInf(gpu_dst, gpu_dstScalar, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInf(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC1", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    gpu::Image<Pixel64fC1> gpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    Pixel64fC1 gpu_res;
    mpp::cuda::DevVar<Pixel64fC1> gpu_dst(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.NormInfMasked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
}

TEST_CASE("64fC2", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    gpu::Image<Pixel64fC2> gpu_src1(size, size);
    Pixel64fC2 cpu_dst;
    Pixel64fC2 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC2> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC3", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    gpu::Image<Pixel64fC3> gpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    Pixel64fC3 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC3> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC4", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    gpu::Image<Pixel64fC4> gpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    Pixel64fC4 gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dst.w == Approx(gpu_res.w).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}

TEST_CASE("64fC4A", "[CUDA.Statistics.NormInfMasked]")
{
    const uint seed                  = Catch::getSeed();
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask   = cpu::Image<Pixel8uC1>::Load(root / "maskAllSizes.tif");
    gpu::Image<Pixel8uC1> gpu_mask(size, size);
    cpu_mask >> gpu_mask;

    cpu::Image<Pixel64fC4A> cpu_src1(size, size);
    gpu::Image<Pixel64fC4A> gpu_src1(size, size);
    Pixel64fC4A cpu_dst;
    Pixel64fC4A gpu_res;
    double cpu_dstScalar;
    double gpu_resScalar;
    mpp::cuda::DevVar<Pixel64fC4A> gpu_dst(1);
    mpp::cuda::DevVar<double> gpu_dstScalar(1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.NormInfMaskedBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    gpu_src1.NormInfMasked(gpu_dst, gpu_dstScalar, gpu_mask, gpu_buffer);
    gpu_dst >> gpu_res;
    gpu_dstScalar >> gpu_resScalar;

    cpu_src1.NormInfMasked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(gpu_res.x).margin(0.00001));
    CHECK(cpu_dst.y == Approx(gpu_res.y).margin(0.00001));
    CHECK(cpu_dst.z == Approx(gpu_res.z).margin(0.00001));
    CHECK(cpu_dstScalar == Approx(gpu_resScalar).margin(0.00001));
}
