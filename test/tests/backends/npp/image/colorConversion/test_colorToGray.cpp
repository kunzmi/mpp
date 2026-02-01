#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC2View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
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

TEST_CASE("8uC3", "[NPP.ColorConversion.ColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.ColorToGray(npp_dst1, Vec3f(0.299f, 0.587f, 0.114f), nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.ColorToGray(cpu_dst1, Vec3f(0.299f, 0.587f, 0.114f));

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
}

TEST_CASE("8uC4", "[NPP.ColorConversion.ColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.ColorToGray(npp_dst1, Vec4f(0.0577f, 0.5205f, 0.0025f, 0.4193f), nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.ColorToGray(cpu_dst1, Vec4f(0.0577f, 0.5205f, 0.0025f, 0.4193f));

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
}

TEST_CASE("16uC3", "[NPP.ColorConversion.ColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(256, 256);
    cpu::Image<Pixel16uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel16uC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image16uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image16uC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.ColorToGray(npp_dst1, Vec3f(0.299f, 0.587f, 0.114f), nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.ColorToGray(cpu_dst1, Vec3f(0.299f, 0.587f, 0.114f));

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
}

TEST_CASE("16uC4", "[NPP.ColorConversion.ColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(256, 256);
    cpu::Image<Pixel16uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel16uC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image16uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image16uC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.ColorToGray(npp_dst1, Vec4f(0.0577f, 0.5205f, 0.0025f, 0.4193f), nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.ColorToGray(cpu_dst1, Vec4f(0.0577f, 0.5205f, 0.0025f, 0.4193f));

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
}

TEST_CASE("32fC3", "[NPP.ColorConversion.ColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(256, 256);
    cpu::Image<Pixel32fC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel32fC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image32fC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image32fC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.ColorToGray(npp_dst1, Vec3f(0.299f, 0.587f, 0.114f), nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.ColorToGray(cpu_dst1, Vec3f(0.299f, 0.587f, 0.114f));

    CHECK(npp_res1.IsSimilar(cpu_dst1, 0.0001f));
}

TEST_CASE("32fC4", "[NPP.ColorConversion.ColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(256, 256);
    cpu::Image<Pixel32fC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel32fC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image32fC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image32fC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.ColorToGray(npp_dst1, Vec4f(0.0577f, 0.5205f, 0.0025f, 0.4193f), nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.ColorToGray(cpu_dst1, Vec4f(0.0577f, 0.5205f, 0.0025f, 0.4193f));

    CHECK(npp_res1.IsSimilar(cpu_dst1, 0.0001f));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.RGBToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.RGBToGray(npp_dst1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.ColorToGray(cpu_dst1, Vec3f(0.299f, 0.587f, 0.114f));

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
}
