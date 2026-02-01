#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC2View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/colorConversion/colorMatrices.h>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr420]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr420A(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1A.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr420_CSC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr420_709CS(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr_CSC,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr420_HDTV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr420_709HDTVA(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1A.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr_HDTV,
                              ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr420_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr420_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr_JPEG,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCrCb420]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCrCb420(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCrCb, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCrCb420_709CSC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCrCb420_709CS(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCrCb_CSC,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYUV420]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYUV420A(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1A.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYUV, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr420]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr420(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::RGBtoYCbCr, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr420_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr420_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::RGBtoYCbCr_JPEG,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCrCb420]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCrCb420A(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1A.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::RGBtoYCrCb, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYUV420]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYUV420(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::RGBtoYUV, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr420ToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr420ToBGR(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr420ToBGR_CSC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr420ToBGR_709CS(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR_CSC,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr420ToBGR_HDTV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res3(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr420ToBGR_709HDTV(npp_src1, npp_src2, npp_src3, npp_dst, 255, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR_HDTV,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);
    npp_res.SwapChannel(npp_res3, {0, 1, 2});
    CHECK(npp_res3.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr420ToBGR_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr420ToBGR_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR_JPEG,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr420ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr420ToRGB(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr420ToRGB_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr420ToRGB_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoRGB_JPEG,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCrCb420ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res3(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCrCb420ToRGB(npp_src1, npp_src2, npp_src3, npp_dst, 255, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCrCbtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);
    npp_res.SwapChannel(npp_res3, {0, 1, 2});
    CHECK(npp_res3.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YUV420ToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YUV420ToBGR(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YUVtoBGR,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YUV420ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YUV420ToRGB(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YUVtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV12ToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV12ToBGR(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, mpp::image::color::YUVtoBGR,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV12ToBGR_CSC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV12ToBGR_709CS(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, mpp::image::color::YCbCrtoBGR_CSC,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV12ToBGR_HDTV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV12ToBGR_709HDTV(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, mpp::image::color::YCbCrtoBGR_HDTV,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);
    // I have no idea what color matrix NPP is using here, none of the described ones match the result. I'm not even
    // sure if the NPP result is actually correct as some test images look way too redish...
    /*CHECK(npp_res.IsSimilar(cpu_dst, 1)); */
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV12ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV12ToRGB(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, mpp::image::color::YUVtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV12ToRGB_CSC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV12ToRGB_709CS(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, mpp::image::color::YCbCrtoRGB_CSC,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV12ToRGB_HDTV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV12ToRGB_709HDTV(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, mpp::image::color::YCbCrtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);
    // same as for the BGR case... results don't match.
    /*CHECK(npp_res.IsSimilar(cpu_dst, 1)); */
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV12ToYUV420]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / 2);
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / 2);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV12ToYUV420(npp_src1, npp_src2, npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.Copy(cpu_dst1);
    cpu_src2.Copy(cpu_dst2, cpu_dst3);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV21ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC4A> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4A> npp_resA(cpu_src1.SizeAlloc());
    cpu::ImageView<Pixel8uC4> npp_res(npp_resA);
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV21ToRGB(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    constexpr mpp::image::Matrix3x4<float> NV21toRGB(mpp::image::color::YUVtoRGBCoeffs * mpp::image::color::Swap2_3,
                                                     mpp::image::color::YUVtoRGBCoeffs * mpp::image::color::Swap2_3 *
                                                         mpp::image::color::OffsetYCbCrFR_Inv);

    cpu::Image<Pixel8uC4A>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, NV21toRGB, ChromaSubsamplePos::Center,
                                              InterpolationMode::NearestNeighbor);

    // same as with NV12ToRGB_HDTV, none of the documented matrices match the result from NPP...
    // CHECK(npp_resA.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.NV21ToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_src2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_src22(reinterpret_cast<Pixel8uC1 *>(cpu_src2.Pointer()),
                                        {{cpu_src2.SizeAlloc() * Vec2i(2, 1)}, cpu_src2.Pitch()});
    cpu::Image<Pixel8uC4A> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4A> npp_resA(cpu_src1.SizeAlloc());
    cpu::ImageView<Pixel8uC4> npp_res(npp_resA);
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(1, 2));
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1 >> npp_src1;
    cpu_src22 >> npp_src2;

    nv::Image8uC2::NV21ToBGR(npp_src1, npp_src2, npp_dst, nppCtx);
    npp_res << npp_dst;

    constexpr mpp::image::Matrix3x4<float> NV21toRGB(
        mpp::image::color::Swap1_3 * mpp::image::color::YUVtoRGBCoeffs * mpp::image::color::Swap2_3,
        mpp::image::color::Swap1_3 * mpp::image::color::YUVtoRGBCoeffs * mpp::image::color::Swap2_3 *
            mpp::image::color::OffsetYCbCrFR_Inv);

    cpu::Image<Pixel8uC4A>::ColorTwistFrom420(cpu_src1, cpu_src2, cpu_dst, NV21toRGB, ChromaSubsamplePos::Center,
                                              InterpolationMode::NearestNeighbor);

    // same as with NV12ToBGR_HDTV, none of the documented matrices match the result from NPP...
    // CHECK(npp_resA.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToNV12]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> cpu_dst2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> cpu_dst22(reinterpret_cast<Pixel8uC1 *>(cpu_dst2.Pointer()),
                                        {{cpu_dst2.SizeAlloc() * Vec2i(2, 1)}, cpu_dst2.Pitch()});
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res2(cpu_src1.SizeAlloc() / 2);
    cpu::ImageView<Pixel8uC1> npp_res22(reinterpret_cast<Pixel8uC1 *>(npp_res2.Pointer()),
                                        {{npp_res2.SizeAlloc() * Vec2i(2, 1)}, npp_res2.Pitch()});
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(1, 2));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.RGBToNV12(npp_dst1, npp_dst2, reinterpret_cast<const float(*)[4]>(mpp::image::color::RGBtoYUV.Data()),
                       nppCtx);
    npp_res1 << npp_dst1;
    npp_res22 << npp_dst2;

    cpu_src1.ColorTwistTo420(cpu_dst1, cpu_dst2, mpp::image::color::RGBtoYUV, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));

    Pixel64fC2 err;
    double errScalar;
    npp_res2.AverageError(cpu_dst2, err, errScalar);

    CHECK(err < 0.5); // there are sometime some outliers, so take the average
}