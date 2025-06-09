#include <backends/cuda/devVar.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <common/defines.h>
#include <common/safeCast.h>
#include <numbers>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Filtering.BilateralGaussFilter]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    const Pixel32fC1 sigmaPos = 15.0f;
    const Pixel32fC1 sigmaVal = 1.5f * 255;

    cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load(root / "bird256bw.tif");
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);
    std::vector<float> posFilter(9 * 9);

    cpu_src1 >> npp_src1;

    npp_src1.FilterBilateralGaussBorder(npp_dst, 5, 1, sigmaVal, sigmaPos, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.PrecomputeBilateralGaussFilter(posFilter.data(), 9, sigmaPos.x);
    cpu_src1.BilateralGaussFilter(cpu_dst, 9, posFilter.data(), sigmaVal.x, BorderType::Replicate);

    Pixel64fC1 maxError;
    Pixel64fC1 meanError;
    cpu_dst.NormDiffInf(npp_res, maxError);
    CHECK(maxError < 8);
    cpu_dst.NormDiffL1(npp_res, meanError);
    meanError /= to_double(cpu_dst.SizeRoi().TotalSize());
    CHECK(meanError < 0.6);
}

TEST_CASE("8uC3", "[NPP.Filtering.BilateralGaussFilter]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    const Pixel32fC1 sigmaPos = 15.0f;
    const Pixel32fC1 sigmaVal = 1.5f * 255;

    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);
    std::vector<float> posFilter(9 * 9);

    cpu_src1 >> npp_src1;

    npp_src1.FilterBilateralGaussBorder(npp_dst, 5, 1, sigmaVal, sigmaPos, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.PrecomputeBilateralGaussFilter(posFilter.data(), 9, sigmaPos.x);
    cpu_src1.BilateralGaussFilter(cpu_dst, 9, posFilter.data(), sigmaVal.x, opp::Norm::L2, BorderType::Replicate);

    // It seems that NPP filters every channel independently...
    Pixel64fC3 maxError;
    Pixel64fC3 meanError;
    double dummy;
    cpu_dst.NormDiffInf(npp_res, maxError, dummy);
    CHECK(maxError < 27);
    cpu_dst.NormDiffL1(npp_res, meanError, dummy);
    meanError /= to_double(cpu_dst.SizeRoi().TotalSize());
    CHECK(meanError < 1.6);
}

TEST_CASE("16uC1", "[NPP.Filtering.BilateralGaussFilter]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    const Pixel32fC1 sigmaPos = 15.0f;
    const Pixel32fC1 sigmaVal = 1.5f * 255;

    cpu::Image<Pixel8uC1> bird = cpu::Image<Pixel8uC1>::Load(root / "bird256bw.tif");

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(size, size);
    std::vector<float> posFilter(9 * 9);

    bird.Convert(cpu_src1);
    cpu_src1 >> npp_src1;

    npp_src1.FilterBilateralGaussBorder(npp_dst, 5, 1, sigmaVal, sigmaPos, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.PrecomputeBilateralGaussFilter(posFilter.data(), 9, sigmaPos.x);
    cpu_src1.BilateralGaussFilter(cpu_dst, 9, posFilter.data(), sigmaVal.x, BorderType::Replicate);

    Pixel64fC1 maxError;
    Pixel64fC1 meanError;
    cpu_dst.NormDiffInf(npp_res, maxError);
    CHECK(maxError < 8);
    cpu_dst.NormDiffL1(npp_res, meanError);
    meanError /= to_double(cpu_dst.SizeRoi().TotalSize());
    CHECK(meanError < 0.6);
}

TEST_CASE("32fC1", "[NPP.Filtering.BilateralGaussFilter]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    const Pixel32fC1 sigmaPos = 15.0f;
    const Pixel32fC1 sigmaVal = 1.5f * 255;

    cpu::Image<Pixel8uC1> bird = cpu::Image<Pixel8uC1>::Load(root / "bird256bw.tif");

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);
    std::vector<float> posFilter(9 * 9);

    bird.Convert(cpu_src1);
    cpu_src1 >> npp_src1;

    npp_src1.FilterBilateralGaussBorder(npp_dst, 5, 1, sigmaVal, sigmaPos, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.PrecomputeBilateralGaussFilter(posFilter.data(), 9, sigmaPos.x);
    cpu_src1.BilateralGaussFilter(cpu_dst, 9, posFilter.data(), sigmaVal.x, BorderType::Replicate);

    Pixel64fC1 maxError;
    Pixel64fC1 meanError;
    cpu_dst.NormDiffInf(npp_res, maxError);
    CHECK(maxError < 7);
    cpu_dst.NormDiffL1(npp_res, meanError);
    meanError /= to_double(cpu_dst.SizeRoi().TotalSize());
    CHECK(meanError < 0.35);
}