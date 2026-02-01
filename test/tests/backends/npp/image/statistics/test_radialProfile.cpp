#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("32fC1", "[NPP.Statistics.CircularRadialProfile]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    nv::Image32fC1 npp_src1(size, size);
    mpp::cuda::DevVar<NppiProfileData> npp_profile(size / 2);
    std::vector<NppiProfileData> res_profile(size / 2);
    std::vector<int> cpu_ProfileCount(size / 2);
    std::vector<Pixel32fC1> cpu_Profile(size / 2);
    std::vector<Pixel32fC1> cpu_ProfileSqr(size / 2);
    constexpr double mean   = 1.0;
    constexpr double stdDev = 3.0;

    cpu_src1.FillRandomNormal(seed, mean, stdDev);

    cpu_src1 >> npp_src1;

    npp_src1.CircularRadialProfile({size / 2, size / 2}, npp_profile.Pointer(), size / 2, nppCtx);
    npp_profile >> res_profile;

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

    int sumCountNPP    = 0;
    float sumMeanNPP   = 0;
    float sumStdDevNPP = 0;
    for (const auto &elem : res_profile)
    {
        sumCountNPP += elem.nPixels;
        sumMeanNPP += elem.nMeanIntensity;
        sumStdDevNPP += elem.nStdDevIntensity;
    }

    // MPP counts every pixel in the circle with radius r, the total number of pixels is thus close to pi*r^2 = ~51471
    // (minus some rounding errors on the edge)
    CHECK(sumCountCPU == 51101);
    CHECK(sumMeanCPU / 128.0f == Approx(mean).margin(0.1));
    CHECK(sumStdDevCPU / 128.0f == Approx(stdDev).margin(0.1));

    // NPP seems to count only pixels that are close to circles of every profile radius step and skips those pixels that
    // fall in-between
    CHECK(sumCountNPP == 32513);
    CHECK(sumMeanNPP / 128.0f == Approx(mean).margin(0.1));
    CHECK(sumStdDevNPP / 128.0f == Approx(stdDev).margin(0.1));
}

TEST_CASE("32fC1", "[NPP.Statistics.EllipticalRadialProfile]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    nv::Image32fC1 npp_src1(size, size);
    mpp::cuda::DevVar<NppiProfileData> npp_profile(size / 2);
    std::vector<NppiProfileData> res_profile(size / 2);
    std::vector<int> cpu_ProfileCount(size / 2);
    std::vector<Pixel32fC1> cpu_Profile(size / 2);
    std::vector<Pixel32fC1> cpu_ProfileSqr(size / 2);
    constexpr double mean   = 1.0;
    constexpr double stdDev = 3.0;

    cpu_src1.FillRandomNormal(seed, mean, stdDev);

    cpu_src1 >> npp_src1;

    npp_src1.EllipticalRadialProfile({size / 2, size / 2}, 2, 0.5f, npp_profile.Pointer(), size / 2, nppCtx);
    npp_profile >> res_profile;

    cpu_src1.RadialProfile(cpu_ProfileCount.data(), cpu_Profile.data(), cpu_ProfileSqr.data(), size / 2,
                           {size / 2, size / 2}, 2, 0.5f);

    // convert sum to mean and sumSqr to StdDev:
    for (size_t i = 0; i < static_cast<size_t>(size / 2); i++)
    {
        Pixel32fC1 std = cpu_ProfileSqr[i] - (cpu_Profile[i] * cpu_Profile[i] / Pixel32fC1(cpu_ProfileCount[i]));
        std /= Pixel32fC1(std::max(1, cpu_ProfileCount[i] - 1));
        std.Sqrt();
        cpu_Profile[i] /= Pixel32fC1(cpu_ProfileCount[i]);
        cpu_ProfileSqr[i] = std;
    }
    cpu_ProfileSqr[0] = 1;
    cpu_Profile[0]    = 1;

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

    int sumCountNPP    = 0;
    float sumMeanNPP   = 0;
    float sumStdDevNPP = 0;
    for (const auto &elem : res_profile)
    {
        sumCountNPP += elem.nPixels;
        sumMeanNPP += elem.nMeanIntensity;
        sumStdDevNPP += elem.nStdDevIntensity;
    }

    CHECK(sumCountCPU == 62396);
    CHECK(sumMeanCPU / 128.0f == Approx(mean).margin(0.1));
    CHECK(sumStdDevCPU / 128.0f == Approx(stdDev).margin(0.1));

    // NPP fails to return any meaningful output?
    CHECK(sumCountNPP == 1);
}
