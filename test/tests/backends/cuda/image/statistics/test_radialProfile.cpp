#include <backends/cuda/devVar.h>
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
#include <common/image/pitchException.h>
#include <common/image/roiException.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace gpu = mpp::image::cuda;

constexpr int size = 256;

TEST_CASE("32fC1", "[NPP.Statistics.CircularRadialProfile]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    std::vector<int> cpu_ProfileCount(size / 2);
    std::vector<Pixel32fC1> cpu_Profile(size / 2);
    std::vector<Pixel32fC1> cpu_ProfileSqr(size / 2);
    std::vector<int> gpu_resProfileCount(size / 2);
    std::vector<Pixel32fC1> gpu_resProfile(size / 2);
    std::vector<Pixel32fC1> gpu_resProfileSqr(size / 2);
    mpp::cuda::DevVar<int> gpu_ProfileCount(size / 2);
    mpp::cuda::DevVar<Pixel32fC1> gpu_Profile(size / 2);
    mpp::cuda::DevVar<Pixel32fC1> gpu_ProfileSqr(size / 2);

    constexpr double mean   = 1.0;
    constexpr double stdDev = 3.0;

    cpu_src1.FillRandomNormal(seed, mean, stdDev);

    cpu_src1 >> gpu_src1;
    gpu_src1.RadialProfile(gpu_ProfileCount, gpu_Profile, gpu_ProfileSqr, {size / 2, size / 2});
    gpu_ProfileCount >> gpu_resProfileCount;
    gpu_Profile >> gpu_resProfile;
    gpu_ProfileSqr >> gpu_resProfileSqr;

    cpu_src1.RadialProfile(cpu_ProfileCount.data(), cpu_Profile.data(), cpu_ProfileSqr.data(), size / 2,
                           {size / 2, size / 2});

    // convert sum to mean and sumSqr to StdDev:
    for (size_t i = 0; i < static_cast<size_t>(size / 2); i++)
    {
        Pixel32fC1 std = cpu_ProfileSqr[i] - (cpu_Profile[i] * cpu_Profile[i] / Pixel32fC1(cpu_ProfileCount[i]));
        std /= Pixel32fC1(std::max(1, cpu_ProfileCount[i] - 1));
        std.Sqrt();
        cpu_Profile[i] /= Pixel32fC1(cpu_ProfileCount[i]);
        cpu_ProfileSqr[i] = std;

        std = gpu_resProfileSqr[i] - (gpu_resProfile[i] * gpu_resProfile[i] / Pixel32fC1(gpu_resProfileCount[i]));
        std /= Pixel32fC1(std::max(1, gpu_resProfileCount[i] - 1));
        std.Sqrt();
        gpu_resProfile[i] /= Pixel32fC1(gpu_resProfileCount[i]);
        gpu_resProfileSqr[i] = std;
    }
    int sumCountCPU = 0;
    for (const auto &elem : cpu_ProfileCount)
    {
        sumCountCPU += elem;
    }

    float sumMeanCPU = 0;
    for (const auto &elem : cpu_Profile)
    {
        sumMeanCPU += elem.x;
    }

    float sumStdDevCPU = 0;
    for (const auto &elem : cpu_ProfileSqr)
    {
        sumStdDevCPU += elem.x;
    }

    int sumCountGPU = 0;
    for (const auto &elem : gpu_resProfileCount)
    {
        sumCountGPU += elem;
    }

    float sumMeanGPU = 0;
    for (const auto &elem : gpu_resProfile)
    {
        sumMeanGPU += elem.x;
    }

    float sumStdDevGPU = 0;
    for (const auto &elem : gpu_resProfileSqr)
    {
        sumStdDevGPU += elem.x;
    }

    // MPP counts every pixel in the circle with radius r, the total number of pixels is thus close to pi*r^2 = ~51471
    // (minus some rounding errors on the edge)
    CHECK(sumCountCPU == 51101);
    CHECK(sumMeanCPU / 128.0f == Approx(mean).margin(0.1));
    CHECK(sumStdDevCPU / 128.0f == Approx(stdDev).margin(0.1));

    CHECK(sumCountGPU == sumCountCPU);
    CHECK(sumMeanGPU == Approx(sumMeanCPU).margin(0.0001));
    CHECK(sumStdDevGPU == Approx(sumStdDevCPU).margin(0.0001));
}

TEST_CASE("32fC1", "[NPP.Statistics.EllipticalRadialProfile]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    std::vector<int> cpu_ProfileCount(size / 2);
    std::vector<Pixel32fC1> cpu_Profile(size / 2);
    std::vector<Pixel32fC1> cpu_ProfileSqr(size / 2);
    std::vector<int> gpu_resProfileCount(size / 2);
    std::vector<Pixel32fC1> gpu_resProfile(size / 2);
    std::vector<Pixel32fC1> gpu_resProfileSqr(size / 2);
    mpp::cuda::DevVar<int> gpu_ProfileCount(size / 2);
    mpp::cuda::DevVar<Pixel32fC1> gpu_Profile(size / 2);
    mpp::cuda::DevVar<Pixel32fC1> gpu_ProfileSqr(size / 2);

    constexpr double mean   = 1.0;
    constexpr double stdDev = 3.0;

    cpu_src1.FillRandomNormal(seed, mean, stdDev);

    cpu_src1 >> gpu_src1;
    gpu_src1.RadialProfile(gpu_ProfileCount, gpu_Profile, gpu_ProfileSqr, {size / 2, size / 2}, 2, 45);
    gpu_ProfileCount >> gpu_resProfileCount;
    gpu_Profile >> gpu_resProfile;
    gpu_ProfileSqr >> gpu_resProfileSqr;

    cpu_src1.RadialProfile(cpu_ProfileCount.data(), cpu_Profile.data(), cpu_ProfileSqr.data(), size / 2,
                           {size / 2, size / 2}, 2, 45);

    // convert sum to mean and sumSqr to StdDev:
    for (size_t i = 0; i < static_cast<size_t>(size / 2); i++)
    {
        Pixel32fC1 std = cpu_ProfileSqr[i] - (cpu_Profile[i] * cpu_Profile[i] / Pixel32fC1(cpu_ProfileCount[i]));
        std /= Pixel32fC1(std::max(1, cpu_ProfileCount[i] - 1));
        std.Sqrt();
        cpu_Profile[i] /= Pixel32fC1(cpu_ProfileCount[i]);
        cpu_ProfileSqr[i] = std;

        std = gpu_resProfileSqr[i] - (gpu_resProfile[i] * gpu_resProfile[i] / Pixel32fC1(gpu_resProfileCount[i]));
        std /= Pixel32fC1(std::max(1, gpu_resProfileCount[i] - 1));
        std.Sqrt();
        gpu_resProfile[i] /= Pixel32fC1(gpu_resProfileCount[i]);
        gpu_resProfileSqr[i] = std;
    }
    int sumCountCPU = 0;
    for (const auto &elem : cpu_ProfileCount)
    {
        sumCountCPU += elem;
    }

    float sumMeanCPU = 0;
    for (const auto &elem : cpu_Profile)
    {
        sumMeanCPU += elem.x;
    }

    float sumStdDevCPU = 0;
    for (const auto &elem : cpu_ProfileSqr)
    {
        sumStdDevCPU += elem.x;
    }

    int sumCountGPU = 0;
    for (const auto &elem : gpu_resProfileCount)
    {
        sumCountGPU += elem;
    }

    float sumMeanGPU = 0;
    for (const auto &elem : gpu_resProfile)
    {
        sumMeanGPU += elem.x;
    }

    float sumStdDevGPU = 0;
    for (const auto &elem : gpu_resProfileSqr)
    {
        sumStdDevGPU += elem.x;
    }

    CHECK(sumCountCPU == 59594);
    CHECK(sumMeanCPU / 128.0f == Approx(mean).margin(0.1));
    CHECK(sumStdDevCPU / 128.0f == Approx(stdDev).margin(0.1));

    CHECK(sumCountGPU == sumCountCPU);
    CHECK(sumMeanGPU == Approx(sumMeanCPU).margin(0.0001));
    CHECK(sumStdDevGPU == Approx(sumStdDevCPU).margin(0.0001));
}
