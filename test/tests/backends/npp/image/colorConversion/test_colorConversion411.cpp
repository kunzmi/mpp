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

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr411]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr411A(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1A.ColorTwistTo411(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.BGRToYCbCr411_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.BGRToYCbCr411_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo411(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::BGRtoYCbCr_JPEG,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr411]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr411(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo411(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::RGBtoYCbCr, ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToYCbCr411_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> cpu_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> cpu_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    cpu::Image<Pixel8uC1> npp_res3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_dst3(cpu_src1.SizeAlloc() / Vec2i(4, 1));

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;
    npp_src1.RGBToYCbCr411_JPEG(npp_dst1, npp_dst2, npp_dst3, nppCtx);
    npp_res1 << npp_dst1;
    npp_res2 << npp_dst2;
    npp_res3 << npp_dst3;

    cpu_src1.ColorTwistTo411(cpu_dst1, cpu_dst2, cpu_dst3, mpp::image::color::RGBtoYCbCr_JPEG,
                             ChromaSubsamplePos::Center);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
    CHECK(npp_res2.IsSimilar(cpu_dst2, 1));
    CHECK(npp_res3.IsSimilar(cpu_dst3, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr411ToBGR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 4, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 4, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr411ToBGR(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom411(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr411ToBGR_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 4, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 4, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr411ToBGR_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom411(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoBGR_JPEG,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr411ToRGB]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 4, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 4, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr411ToRGB(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom411(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoRGB,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.YCbCr411ToRGB_JPEG]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_src2(256 / 4, 256);
    cpu::Image<Pixel8uC1> cpu_src3(256 / 4, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src2(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC1 npp_src3(cpu_src1.SizeAlloc() / Vec2i(4, 1));
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src3.FillRandom(seed + 2);
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_src3 >> npp_src3;
    nv::Image8uC3::YCbCr411ToRGB_JPEG(npp_src1, npp_src2, npp_src3, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::ColorTwistFrom411(cpu_src1, cpu_src2, cpu_src3, cpu_dst, mpp::image::color::YCbCrtoRGB_JPEG,
                                             ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}