#include <backends/npp/image/image16f.h>
#include <backends/npp/image/image16fC1View.h>
#include <backends/npp/image/image16fC2View.h>
#include <backends/npp/image/image16fC3View.h>
#include <backends/npp/image/image16fC4View.h>
#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC2View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
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
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/safeCast.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    // NPP returns 0 for source values > 180 and 255 for values <= 180 from saturation on. No idea why this is, but to
    // avoid issues we clamp the values to max 180.
    for (auto &pixel : cpu_src1)
    {
        pixel.Value().x = to_byte(pixel.Value().x % 180);
    }

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Sqr();
    npp_src1.Sqr(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    // NPP returns 0 for source values > 180 and 255 for values <= 180 from saturation on. No idea why this is, but to
    // avoid issues we clamp the values to max 180.
    for (auto &pixel : cpu_src1)
    {
        pixel.Value().x = to_byte(pixel.Value().x % 180);
        pixel.Value().y = to_byte(pixel.Value().y % 180);
        pixel.Value().z = to_byte(pixel.Value().z % 180);
    }

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Sqr();
    npp_src1.Sqr(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    // NPP returns 0 for source values > 180 and 255 for values <= 180 from saturation on. No idea why this is, but to
    // avoid issues we clamp the values to max 180.
    for (auto &pixel : cpu_src1)
    {
        pixel.Value().x = to_byte(pixel.Value().x % 180);
        pixel.Value().y = to_byte(pixel.Value().y % 180);
        pixel.Value().z = to_byte(pixel.Value().z % 180);
        pixel.Value().w = to_byte(pixel.Value().w % 180);
    }

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Sqr();
    npp_src1.Sqr(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.Sqr]")
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

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);

    // NPP returns 0 for source values > 180 and 255 for values <= 180 from saturation on. No idea why this is, but to
    // avoid issues we clamp the values to max 180.
    for (auto &pixel : cpu_src1)
    {
        pixel.Value().x = to_byte(pixel.Value().x % 180);
        pixel.Value().y = to_byte(pixel.Value().y % 180);
        pixel.Value().z = to_byte(pixel.Value().z % 180);
        pixel.Value().w = 255;
    }

    cpu_src1 >> npp_src1;

    cpu_src1A.Sqr(cpu_dstA);
    npp_src1.SqrA(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Sqr();
    npp_src1.SqrA(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.Sqr]")
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

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1A.Sqr(cpu_dstA);
    npp_src1.SqrA(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Sqr();
    npp_dst.SqrA(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC3", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC3::GetStreamContext();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> npp_res(size, size);
    nv::Image16fC3 npp_src1(size, size);
    nv::Image16fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC4", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC4::GetStreamContext();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> npp_res(size, size);
    nv::Image16fC4 npp_src1(size, size);
    nv::Image16fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Sqr(cpu_dst);
    npp_src1.Sqr(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Sqr();
    npp_dst.Sqr(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Arithmetic.Sqr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel32fC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1A.Sqr(cpu_dstA);
    npp_src1.SqrA(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Sqr();
    npp_dst.SqrA(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}