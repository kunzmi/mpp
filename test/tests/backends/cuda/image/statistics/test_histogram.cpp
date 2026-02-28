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

TEST_CASE("8uC1", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    std::vector<Pixel32sC1> cpu_dst(histSize, 0);
    std::vector<int> gpu_res(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst(histSize);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst << gpu_res; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i] == gpu_res[i]);
    }
}

TEST_CASE("8uC2", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    std::vector<Pixel32sC2> cpu_dst(histSize, 0);
    std::vector<int> gpu_res1(histSize, 0);
    std::vector<int> gpu_res2(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst1(histSize);
    mpp::cuda::DevVar<int> gpu_dst2(histSize);
    mpp::cuda::DevVarView<int> gpu_dst[] = {gpu_dst1, gpu_dst2};
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst1 << gpu_res1; // set everything to 0
    gpu_dst2 << gpu_res2; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst1 >> gpu_res1;
    gpu_dst2 >> gpu_res2;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i].x == gpu_res1[i]);
        CHECK(cpu_dst[i].y == gpu_res2[i]);
    }
}

TEST_CASE("8uC3", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    std::vector<Pixel32sC3> cpu_dst(histSize, 0);
    std::vector<int> gpu_res1(histSize, 0);
    std::vector<int> gpu_res2(histSize, 0);
    std::vector<int> gpu_res3(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst1(histSize);
    mpp::cuda::DevVar<int> gpu_dst2(histSize);
    mpp::cuda::DevVar<int> gpu_dst3(histSize);
    mpp::cuda::DevVarView<int> gpu_dst[] = {gpu_dst1, gpu_dst2, gpu_dst3};
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst1 << gpu_res1; // set everything to 0
    gpu_dst2 << gpu_res2; // set everything to 0
    gpu_dst3 << gpu_res3; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst1 >> gpu_res1;
    gpu_dst2 >> gpu_res2;
    gpu_dst3 >> gpu_res3;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i].x == gpu_res1[i]);
        CHECK(cpu_dst[i].y == gpu_res2[i]);
        CHECK(cpu_dst[i].z == gpu_res3[i]);
    }
}

TEST_CASE("8uC4", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    std::vector<Pixel32sC4> cpu_dst(histSize, 0);
    std::vector<int> gpu_res1(histSize, 0);
    std::vector<int> gpu_res2(histSize, 0);
    std::vector<int> gpu_res3(histSize, 0);
    std::vector<int> gpu_res4(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst1(histSize);
    mpp::cuda::DevVar<int> gpu_dst2(histSize);
    mpp::cuda::DevVar<int> gpu_dst3(histSize);
    mpp::cuda::DevVar<int> gpu_dst4(histSize);
    mpp::cuda::DevVarView<int> gpu_dst[] = {gpu_dst1, gpu_dst2, gpu_dst3, gpu_dst4};
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst1 << gpu_res1; // set everything to 0
    gpu_dst2 << gpu_res2; // set everything to 0
    gpu_dst3 << gpu_res3; // set everything to 0
    gpu_dst4 << gpu_res4; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst1 >> gpu_res1;
    gpu_dst2 >> gpu_res2;
    gpu_dst3 >> gpu_res3;
    gpu_dst4 >> gpu_res4;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i].x == gpu_res1[i]);
        CHECK(cpu_dst[i].y == gpu_res2[i]);
        CHECK(cpu_dst[i].z == gpu_res3[i]);
        CHECK(cpu_dst[i].w == gpu_res4[i]);
    }
}

TEST_CASE("8uC4A", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    std::vector<Pixel32sC4A> cpu_dst(histSize, 0);
    std::vector<int> gpu_res1(histSize, 0);
    std::vector<int> gpu_res2(histSize, 0);
    std::vector<int> gpu_res3(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst1(histSize);
    mpp::cuda::DevVar<int> gpu_dst2(histSize);
    mpp::cuda::DevVar<int> gpu_dst3(histSize);
    mpp::cuda::DevVarView<int> gpu_dst[] = {gpu_dst1, gpu_dst2, gpu_dst3};
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst1 << gpu_res1; // set everything to 0
    gpu_dst2 << gpu_res2; // set everything to 0
    gpu_dst3 << gpu_res3; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst1 >> gpu_res1;
    gpu_dst2 >> gpu_res2;
    gpu_dst3 >> gpu_res3;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i].x == gpu_res1[i]);
        CHECK(cpu_dst[i].y == gpu_res2[i]);
        CHECK(cpu_dst[i].z == gpu_res3[i]);
    }
}

TEST_CASE("16fC1", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    std::vector<Pixel32sC1> cpu_dst(histSize, 0);
    std::vector<int> gpu_res(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst(histSize);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst << gpu_res; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i] == gpu_res[i]);
    }
}

TEST_CASE("16bfC1", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel16bfC1> cpu_src1(size, size);
    gpu::Image<Pixel16bfC1> gpu_src1(size, size);
    std::vector<Pixel32sC1> cpu_dst(histSize, 0);
    std::vector<int> gpu_res(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst(histSize);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst << gpu_res; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i] == gpu_res[i]);
    }
}

TEST_CASE("32fC1", "[NPP.Statistics.HistogramEven]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    std::vector<Pixel32sC1> cpu_dst(histSize, 0);
    std::vector<int> gpu_res(histSize, 0);
    mpp::cuda::DevVar<int> gpu_dst(histSize);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst << gpu_res; // set everything to 0

    gpu_src1.HistogramEven(gpu_dst, 0, 256, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 256);

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i] == gpu_res[i]);
    }
}

TEST_CASE("32fC1", "[NPP.Statistics.HistogramRange]")
{
    const uint seed        = Catch::getSeed();
    constexpr int histSize = 16;

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    std::vector<Pixel32sC1> cpu_dst(histSize, 0);
    std::vector<int> gpu_res(histSize, 0);
    std::vector<float> levels(histSize + 1);
    mpp::cuda::DevVar<int> gpu_dst(histSize);
    mpp::cuda::DevVar<float> gpu_levels(histSize + 1);
    mpp::cuda::DevVar<byte> gpu_buffer(gpu_src1.HistogramEvenBufferSize(histSize + 1));

    for (size_t i = 0; i < static_cast<size_t>(histSize) + 1; i++)
    {
        levels[i] = static_cast<float>(i * i) / static_cast<float>(histSize * histSize);
    }
    gpu_levels << levels;

    cpu_src1.FillRandom(seed);
    cpu_src1 >> gpu_src1;
    gpu_dst << gpu_res; // set everything to 0

    gpu_src1.HistogramRange(gpu_dst, gpu_levels, gpu_buffer);
    gpu_dst >> gpu_res;

    cpu_src1.HistogramRange(cpu_dst.data(), histSize, reinterpret_cast<Pixel32fC1 *>(levels.data()));

    for (size_t i = 0; i < histSize; i++)
    {
        CHECK(cpu_dst[i] == gpu_res[i]);
    }
}