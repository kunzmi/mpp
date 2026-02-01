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

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToCbYCr422]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToCbYCr422A(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.ColorTwistTo422(cpu_dst, mpp::image::color::BGRtoYCbCr, ChromaSubsamplePos::Center, true);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToCbYCr422_709HDTV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToCbYCr422_709HDTV(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistTo422(cpu_dst, mpp::image::color::BGRtoYCbCr_HDTV, ChromaSubsamplePos::Center, true);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr422]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr422(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistTo422(cpu_dst, mpp::image::color::BGRtoYCbCr, ChromaSubsamplePos::Center, false);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr422P3]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr422(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo422(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr422_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr422_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo422(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr_JPEG,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.CbYCr422ToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC2> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res2(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.CbYCr422ToBGR(npp_dst, 255, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistFrom422(cpu_dst, mpp::image::color::YCbCrtoBGR, true);
    npp_res.SwapChannel(npp_res2, {0, 1, 2});
    CHECK(npp_res2.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.CbYCr422ToBGR_709HDTV]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC2> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.CbYCr422ToBGR_709HDTV(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistFrom422(cpu_dst, mpp::image::color::YCbCrtoBGR_HDTV, true);
    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.CbYCr422ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC2> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.CbYCr422ToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistFrom422(cpu_dst, mpp::image::color::YCbCrtoRGB, true);
    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.nppiRGBToCbYCr422]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToCbYCr422(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistTo422(cpu_dst, mpp::image::color::RGBtoYCbCr, ChromaSubsamplePos::Center, true);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr422]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr422(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistTo422(cpu_dst, mpp::image::color::RGBtoYCbCr, ChromaSubsamplePos::Center, false);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr422_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr422_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo422(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::RGBtoYCbCr_JPEG,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCrCb422]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCrCb422(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistTo422(cpu_dst, mpp::image::color::RGBtoYCrCb, ChromaSubsamplePos::Center, false);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYUV422]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYUV422(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistTo422(cpu_dst, mpp::image::color::RGBtoYUV, ChromaSubsamplePos::Center, false);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr422ToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 2, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 2, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr422ToBGR(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom422(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr422ToBGR_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 2, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 2, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr422ToBGR_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom422(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR_JPEG,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr422ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 2, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 2, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr422ToRGB(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom422(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr422ToRGB_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 2, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 2, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr422ToRGB_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom422(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoRGB_JPEG,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCrCb422ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC2> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC2 npp_src(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src;
    npp_src.YCrCb422ToRGB(npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ColorTwistFrom422(cpu_dst, mpp::image::color::YCrCbtoRGB, false);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YUV422ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 2, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 2, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YUV422ToRGB(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom422(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YUVtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.ConvertSampling422P3C2]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> cpu_src3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC2> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC2> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC2 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr422(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ConvertSampling422(cpu_src1, cpu_src2, cpu_src3, cpu_dst, false);

    CHECK(npp_res.IsIdentical(cpu_dst));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.ConvertSampling422C2P3]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC2> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC2 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(2, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(2, 1));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    npp_src1.YCbCr422(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ConvertSampling422(cpu_dst1, cpu_dst2, cpu_dst3, false);

    CHECK(npp_res1.IsIdentical(cpu_dst1));
    CHECK(npp_res2.IsIdentical(cpu_dst2));
    CHECK(npp_res3.IsIdentical(cpu_dst3));
}