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

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToHLS]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToHLS(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;
    cpu::Image<Pixel8uC3>::Copy(npp_res1, npp_res2, npp_res3, npp_res);
    cpu_src1.BGRtoHLS(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToLab]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToLab(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.BGRtoLab(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;
    cpu::Image<Pixel8uC3>::Copy(npp_res1, npp_res2, npp_res3, npp_res);
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::BGRtoYCbCr);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr444_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr444_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;
    cpu::Image<Pixel8uC3>::Copy(npp_res1, npp_res2, npp_res3, npp_res);
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::BGRtoYCbCr_JPEG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYUV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYUV(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::BGRtoYUV);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.HLSToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.HLSToBGR(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;
    cpu::Image<Pixel8uC3>::Copy(npp_res1, npp_res2, npp_res3, npp_res);
    cpu_src1.HLStoBGR(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.HLSToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.HLSToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.HLStoRGB(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.HSVToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.HSVToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.HSVtoRGB(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.LUVToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.LUVToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.LUVtoRGB(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.LabToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.LabToBGR(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.LabtoBGR(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToHLS]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToHLS(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.RGBtoHLS(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToHSV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToHSV(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.RGBtoHSV(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToLUV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToLUV(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.RGBtoLUV(cpu_dst, 255.0f);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToXYZ]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToXYZ(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::RGBtoXYZ);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYC(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::RGBtoYCC);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::RGBtoYCbCr);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr444_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr444_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;
    cpu::Image<Pixel8uC3>::Copy(npp_res1, npp_res2, npp_res3, npp_res);
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::RGBtoYCbCr_JPEG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYUV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYUV(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::RGBtoYUV);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.XYZToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.XYZToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::XYZtoRGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCCToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.YCCToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwist(cpu_dst, mpp ::image::color::YCCtoRGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr444ToBGR_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src(256, 256);
    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src.Copy(cpu_src1, cpu_src2, cpu_src3);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr444ToBGR_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src.ColorTwist(cpu_dst, mpp::image::color::YCbCrtoBGR_JPEG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr444ToRGB_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src(256, 256);
    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src.Copy(cpu_src1, cpu_src2, cpu_src3);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr444ToRGB_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src.ColorTwist(cpu_dst, mpp::image::color::YCbCrtoRGB_JPEG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCrToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src(256, 256);
    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src.Copy(cpu_src1, cpu_src2, cpu_src3);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCrToBGR(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src.ColorTwist(cpu_dst, mpp::image::color::YCbCrtoBGR);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCrToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src(256, 256);
    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src.Copy(cpu_src1, cpu_src2, cpu_src3);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCrToRGB(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src.ColorTwist(cpu_dst, mpp::image::color::YCbCrtoRGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCrToBGR_709CSC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src(256, 256);
    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src.Copy(cpu_src1, cpu_src2, cpu_src3);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCrToBGR_709CS(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src.ColorTwist(cpu_dst, mpp::image::color::YCbCrtoBGR_CSC);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YUVToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.YUVToBGR(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::YUVtoBGR);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YUVToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.YUVToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;
    cpu_src1.ColorTwist(cpu_dst, mpp::image::color::YUVtoRGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}