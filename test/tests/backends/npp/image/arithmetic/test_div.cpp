#include <backends/npp/image/image16f.h>
#include <backends/npp/image/image16fC1View.h>
#include <backends/npp/image/image16fC2View.h>
#include <backends/npp/image/image16fC3View.h>
#include <backends/npp/image/image16fC4View.h>
#include <backends/npp/image/image16sc.h>
#include <backends/npp/image/image16scC1View.h>
#include <backends/npp/image/image16scC2View.h>
#include <backends/npp/image/image16scC3View.h>
#include <backends/npp/image/image16scC4View.h>
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
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fcC1View.h>
#include <backends/npp/image/image32fcC2View.h>
#include <backends/npp/image/image32fcC3View.h>
#include <backends/npp/image/image32fcC4View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32sc.h>
#include <backends/npp/image/image32scC1View.h>
#include <backends/npp/image/image32scC2View.h>
#include <backends/npp/image/image32scC3View.h>
#include <backends/npp/image/image32scC4View.h>
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

TEST_CASE("8uC1", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_src2(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2);
    npp_src2.Div(npp_src1, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -1);
    npp_dst.Div(npp_src2, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_src2(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2);
    npp_src2.Div(npp_src1, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -1);
    npp_dst.Div(npp_src2, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2);
    npp_src2.Div(npp_src1, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -1);
    npp_dst.Div(npp_src2, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    nv::Image8uC4 npp_dst(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel8uC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel8uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Div(cpu_src2A, cpu_dstA);
    npp_src2.DivA(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, -2);
    npp_src2.DivA(npp_src1, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, cpu_dstA, -1);
    npp_dst.DivA(npp_src2, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_src2(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -8);
    npp_src2.Div(npp_src1, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -8);
    npp_dst.Div(npp_src2, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_src2(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -8);
    npp_src2.Div(npp_src1, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -8);
    npp_dst.Div(npp_src2, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -8);
    npp_src2.Div(npp_src1, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -8);
    npp_dst.Div(npp_src2, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    nv::Image16uC4 npp_dst(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16uC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel16uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Div(cpu_src2A, cpu_dstA);
    npp_src2.DivA(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, -8);
    npp_src2.DivA(npp_src1, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, cpu_dstA, -8);
    npp_dst.DivA(npp_src2, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC1", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_src2(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> npp_res(size, size);
    nv::Image32sC1 npp_src1(size, size);
    nv::Image32sC1 npp_src2(size, size);
    nv::Image32sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // here the rounding mode for NPP is different than for 8u/16u
    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2);
    npp_src2.Div(npp_src1, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    // NPP handles denormalized values differently for each variant, here we ignore the difference and just set pixels
    // with division by 0 to the same value to pass the test:

    auto iterSrc2 = cpu_src2.begin();
    auto iterCpu  = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (iterSrc2.Value().x == 0)
        {
            elem.Value()    = 0;
            iterCpu.Value() = 0;
        }
        ++iterSrc2;
        ++iterCpu;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -8);
    npp_dst.Div(npp_src2, -8, nppCtx);

    npp_res << npp_dst;
    auto iterSrc22 = cpu_src2.begin();
    auto iterCpu2  = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (iterSrc22.Value().x == 0)
        {
            elem.Value()     = 0;
            iterCpu2.Value() = 0;
        }
        ++iterSrc22;
        ++iterCpu2;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC3", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC3::GetStreamContext();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_src2(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> npp_res(size, size);
    nv::Image32sC3 npp_src1(size, size);
    nv::Image32sC3 npp_src2(size, size);
    nv::Image32sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // here the rounding mode for NPP is different than for 8u/16u
    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2);
    npp_src2.Div(npp_src1, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    // NPP handles denormalized values differently for each variant, here we ignore the difference and just set pixels
    // with division by 0 to the same value to pass the test:

    auto iterSrc2 = cpu_src2.begin();
    auto iterCpu  = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (iterSrc2.Value().x == 0)
        {
            elem.Value().x    = 0;
            iterCpu.Value().x = 0;
        }
        if (iterSrc2.Value().y == 0)
        {
            elem.Value().y    = 0;
            iterCpu.Value().y = 0;
        }
        if (iterSrc2.Value().z == 0)
        {
            elem.Value().z    = 0;
            iterCpu.Value().z = 0;
        }
        ++iterSrc2;
        ++iterCpu;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -8);
    npp_dst.Div(npp_src2, -8, nppCtx);

    npp_res << npp_dst;
    auto iterSrc22 = cpu_src2.begin();
    auto iterCpu2  = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (iterSrc22.Value().x == 0)
        {
            elem.Value().x     = 0;
            iterCpu2.Value().x = 0;
        }
        if (iterSrc22.Value().y == 0)
        {
            elem.Value().y     = 0;
            iterCpu2.Value().y = 0;
        }
        if (iterSrc22.Value().z == 0)
        {
            elem.Value().z     = 0;
            iterCpu2.Value().z = 0;
        }
        ++iterSrc22;
        ++iterCpu2;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_src2(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_src2(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2);
    npp_dst.Div(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC3", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC3::GetStreamContext();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_src2(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> npp_res(size, size);
    nv::Image16fC3 npp_src1(size, size);
    nv::Image16fC3 npp_src2(size, size);
    nv::Image16fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2);
    npp_dst.Div(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC4", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC4::GetStreamContext();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_src2(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> npp_res(size, size);
    nv::Image16fC4 npp_src1(size, size);
    nv::Image16fC4 npp_src2(size, size);
    nv::Image16fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2);
    npp_dst.Div(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2);
    npp_dst.Div(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_src2(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2);
    npp_dst.Div(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_src2(size, size);
    nv::Image32fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    npp_src2.Div(npp_src1, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2);
    npp_dst.Div(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_src2(size, size);
    nv::Image32fC4 npp_dst(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel32fC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel32fC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Div(cpu_src2A, cpu_dstA);
    npp_src2.DivA(npp_src1, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, cpu_dstA);
    npp_dst.DivA(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC1", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_src2(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> npp_res(size, size);
    nv::Image16scC1 npp_src1(size, size);
    nv::Image16scC1 npp_src2(size, size);
    nv::Image16scC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1(0, 0) = Pixel16scC1({0, 0});
    cpu_src2(0, 0) = Pixel16scC1({0, 0});
    cpu_src1(1, 0) = Pixel16scC1({1, 1});
    cpu_src2(1, 0) = Pixel16scC1({0, 0});

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // here the rounding mode for NPP is different than for 8u/16u non-complex
    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1)); // sometimes small rounding errors occur, so allow a difference of 1

    cpu_src1.Div(cpu_src2, cpu_dst, -8);
    npp_src2.Div(npp_src1, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_dst.Div(cpu_src2, -8);
    npp_dst.Div(npp_src2, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16scC3", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC3::GetStreamContext();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_src2(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> npp_res(size, size);
    nv::Image16scC3 npp_src1(size, size);
    nv::Image16scC3 npp_src2(size, size);
    nv::Image16scC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.Div(cpu_src2, cpu_dst, -8);
    npp_src2.Div(npp_src1, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_dst.Div(cpu_src2, -8);
    npp_dst.Div(npp_src2, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16scC4", "[NPP.Arithmetic.Div]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC4::GetStreamContext();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_src2(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> npp_res(size, size);
    nv::Image16scC4 npp_src1(size, size);
    nv::Image16scC4 npp_src2(size, size);
    nv::Image16scC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.DivA(npp_src1, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.Div(cpu_src2, cpu_dst, -8);
    npp_src2.DivA(npp_src1, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_dst.Div(cpu_src2, 1);
    npp_dst.DivA(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC1", "[NPP.Arithmetic.DivC]")
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

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -2);
    npp_src1.Div(constVal, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -1);
    npp_dst.Div(constVal, npp_dst, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC3> op(seed + 1);
    Pixel8uC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -2);
    npp_src1.Div(constVal, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -1);
    npp_dst.Div(constVal, npp_dst, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -2);
    npp_src1.Div(constVal, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -1);
    npp_dst.Div(constVal, npp_dst, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.DivC]")
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
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.Div(constVal, cpu_dstA);
    npp_src1.DivA(constVal.XYZ(), npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(constVal, cpu_dstA, -2);
    npp_src1.DivA(constVal.XYZ(), npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(constVal, cpu_dstA, -1);
    npp_dst.DivA(constVal.XYZ(), npp_dst, -1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.DivC]")
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

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -8);
    npp_src1.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC3> op(seed + 1);
    Pixel16uC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -8);
    npp_src1.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -8);
    npp_src1.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.DivC]")
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
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.Div(constVal, cpu_dstA);
    npp_src1.DivA(constVal.XYZ(), npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(constVal, cpu_dstA, -8);
    npp_src1.DivA(constVal.XYZ(), npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(constVal, cpu_dstA, -8);
    npp_dst.DivA(constVal.XYZ(), npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC1", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch ::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> npp_res(size, size);
    nv::Image32sC1 npp_src1(size, size);
    nv::Image32sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -2);
    npp_src1.Div(constVal, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    // NPP handles denormalized values differently for each variant, here we ignore the difference and just set pixels
    // with division by 0 to the same value to pass the test:

    auto iterCpu = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (constVal.x == 0)
        {
            elem.Value()    = 0;
            iterCpu.Value() = 0;
        }
        ++iterCpu;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;
    auto iterCpu2 = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (constVal.x == 0)
        {
            elem.Value()     = 0;
            iterCpu2.Value() = 0;
        }
        ++iterCpu2;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC3", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC3::GetStreamContext();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> npp_res(size, size);
    nv::Image32sC3 npp_src1(size, size);
    nv::Image32sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC3> op(seed + 1);
    Pixel32sC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(constVal, cpu_dst, -2);
    npp_src1.Div(constVal, npp_dst, -2, nppCtx);

    npp_res << npp_dst;

    // NPP handles denormalized values differently for each variant, here we ignore the difference and just set pixels
    // with division by 0 to the same value to pass the test:

    auto iterCpu = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (constVal.x == 0)
        {
            elem.Value().x    = 0;
            iterCpu.Value().x = 0;
        }
        if (constVal.y == 0)
        {
            elem.Value().y    = 0;
            iterCpu.Value().y = 0;
        }
        if (constVal.z == 0)
        {
            elem.Value().z    = 0;
            iterCpu.Value().z = 0;
        }
        ++iterCpu;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;
    auto iterCpu2 = cpu_dst.begin();
    for (auto &elem : npp_res)
    {
        if (constVal.x == 0)
        {
            elem.Value().x     = 0;
            iterCpu2.Value().x = 0;
        }
        if (constVal.y == 0)
        {
            elem.Value().y     = 0;
            iterCpu2.Value().y = 0;
        }
        if (constVal.z == 0)
        {
            elem.Value().z     = 0;
            iterCpu2.Value().z = 0;
        }
        ++iterCpu2;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst);
    npp_dst.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC3", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC3::GetStreamContext();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> npp_res(size, size);
    nv::Image16fC3 npp_src1(size, size);
    nv::Image16fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC3> op(seed + 1);
    Pixel16fC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst);
    npp_dst.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC4", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC4::GetStreamContext();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> npp_res(size, size);
    nv::Image16fC4 npp_src1(size, size);
    nv::Image16fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC4> op(seed + 1);
    Pixel16fC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst);
    npp_dst.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst);
    npp_dst.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC3> op(seed + 1);
    Pixel32fC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst);
    npp_dst.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst);
    npp_src1.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(constVal, cpu_dst);
    npp_dst.Div(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Arithmetic.DivC]")
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
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.Div(constVal, cpu_dstA);
    npp_src1.DivA(constVal.XYZ(), npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(constVal, cpu_dstA);
    npp_dst.DivA(constVal.XYZ(), npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC1", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> npp_res(size, size);
    nv::Image16scC1 npp_src1(size, size);
    nv::Image16scC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC1> op(seed + 1);
    Pixel16scC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -8);
    npp_src1.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16scC3", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC3::GetStreamContext();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> npp_res(size, size);
    nv::Image16scC3 npp_src1(size, size);
    nv::Image16scC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC3> op(seed + 1);
    Pixel16scC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src1.Div(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -8);
    npp_src1.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.Div(constVal, npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16scC4", "[NPP.Arithmetic.DivC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC4::GetStreamContext();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> npp_res(size, size);
    nv::Image16scC4 npp_src1(size, size);
    nv::Image16scC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC4> op(seed + 1);
    Pixel16scC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Div(constVal, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src1.DivA(constVal.XYZ(), npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -8);
    npp_src1.DivA(constVal.XYZ(), npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 1));

    cpu_dst.Div(constVal, cpu_dst, -8);
    npp_dst.DivA(constVal.XYZ(), npp_dst, -8, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}