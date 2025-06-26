#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC2View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC2View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    npp_src1.AlphaPremul(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul(constVal.x);
    npp_src1.AlphaPremul(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    npp_src1.AlphaPremul(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul(constVal.x);
    npp_src1.AlphaPremul(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    npp_src1.AlphaPremul(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul(constVal.x);
    npp_src1.AlphaPremul(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel8uC4A> cpu_dstA  = cpu_dst;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.AlphaPremul(constVal.x, cpu_dstA);
    npp_src1.AlphaPremulA(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1A.AlphaPremul(constVal.x);
    npp_src1.AlphaPremulA(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1A.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.AlphaPremulAlphaChannel]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel8uC4A> cpu_dstA  = cpu_dst;

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(cpu_dst);
    npp_src1.AlphaPremulA(npp_dst, nppCtx);

    npp_res << npp_dst;

    // NPP contains a bug: when alpha value is 255, the destination image is not set to 255 but 0 if the tupel-kernel is
    // used. The naive kernel correctly sets the value to 255...
    auto iter_cpu = cpu_dst.begin();
    for (auto &npp_pixel : npp_res)
    {
        if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
        {
            npp_pixel.Value().w = 255;
        }
        ++iter_cpu;
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul();
    npp_src1.AlphaPremulA(nppCtx);

    npp_res << npp_src1;
    auto iter_cpu2 = cpu_dst.begin();
    for (auto &npp_pixel : npp_res)
    {
        if (npp_pixel.Value().w == 0 && iter_cpu2.Value().w == 255)
        {
            npp_pixel.Value().w = 255;
        }
        ++iter_cpu2;
    }
    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    npp_src1.AlphaPremul(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul(constVal.x);
    npp_src1.AlphaPremul(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    npp_src1.AlphaPremul(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul(constVal.x);
    npp_src1.AlphaPremul(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(constVal.x, cpu_dst);
    npp_src1.AlphaPremul(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul(constVal.x);
    npp_src1.AlphaPremul(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.AlphaPremul]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16uC4A> cpu_dstA  = cpu_dst;

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.AlphaPremul(constVal.x, cpu_dstA);
    npp_src1.AlphaPremulA(constVal.x, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1A.AlphaPremul(constVal.x);
    npp_src1.AlphaPremulA(constVal.x, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1A.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.AlphaPremulAlphaChannel]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16uC4A> cpu_dstA  = cpu_dst;

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.AlphaPremul(cpu_dst);
    npp_src1.AlphaPremulA(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.AlphaPremul();
    npp_src1.AlphaPremulA(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsSimilar(npp_res, 1));
}