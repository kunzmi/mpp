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
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image8s.h>
#include <backends/npp/image/image8sC1View.h>
#include <backends/npp/image/image8sC2View.h>
#include <backends/npp/image/image8sC3View.h>
#include <backends/npp/image/image8sC4View.h>
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

TEST_CASE("8uC1", "[NPP.Arithmetic.RShift]")
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
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.RShift]")
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
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.RShift]")
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
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.RShift]")
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

    cpu_dst.Set({127272727});
    npp_dst.Set({127272727}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    cpu_src1A.RShift(constVal, cpu_dstA);
    npp_src1.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.RShift(constVal, cpu_dstA);
    npp_dst.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8sC1", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel8sC1> cpu_dst(size, size);
    cpu::Image<Pixel8sC1> npp_res(size, size);
    nv::Image8sC1 npp_src1(size, size);
    nv::Image8sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8sC3", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC3::GetStreamContext();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel8sC3> cpu_dst(size, size);
    cpu::Image<Pixel8sC3> npp_res(size, size);
    nv::Image8sC3 npp_src1(size, size);
    nv::Image8sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    // NPP code is wrong and not consistent with inplace
    /*cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));*/

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8sC4", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC4::GetStreamContext();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_dst(size, size);
    cpu::Image<Pixel8sC4> npp_res(size, size);
    nv::Image8sC4 npp_src1(size, size);
    nv::Image8sC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    // NPP code is wrong and not consistent with inplace
    /*cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));*/

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8sC4A", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC4::GetStreamContext();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_dst(size, size);
    cpu::Image<Pixel8sC4> npp_res(size, size);
    nv::Image8sC4 npp_src1(size, size);
    nv::Image8sC4 npp_dst(size, size);
    cpu::ImageView<Pixel8sC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel8sC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127272727});
    npp_dst.Set({127272727}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 8);

    cpu_src1 >> npp_src1;

    // NPP code is wrong and not consistent with inplace
    /*cpu_src1A.RShift(constVal, cpu_dstA);
    npp_src1.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));*/

    cpu_dstA.RShift(constVal, cpu_dstA);
    npp_dst.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.RShift]")
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

    cpu_dst.Set({127272727});
    npp_dst.Set({127272727}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1A.RShift(constVal, cpu_dstA);
    npp_src1.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.RShift(constVal, cpu_dstA);
    npp_dst.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC1", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC3", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> npp_res(size, size);
    nv::Image16sC3 npp_src1(size, size);
    nv::Image16sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC4", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC4A", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_dst(size, size);
    cpu::ImageView<Pixel16sC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16sC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127272727});
    npp_dst.Set({127272727}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 16);

    cpu_src1 >> npp_src1;

    cpu_src1A.RShift(constVal, cpu_dstA);
    npp_src1.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.RShift(constVal, cpu_dstA);
    npp_dst.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC1", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> npp_res(size, size);
    nv::Image32sC1 npp_src1(size, size);
    nv::Image32sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 32);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC3", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC3::GetStreamContext();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> npp_res(size, size);
    nv::Image32sC3 npp_src1(size, size);
    nv::Image32sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 32);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC4", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC4::GetStreamContext();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> npp_res(size, size);
    nv::Image32sC4 npp_src1(size, size);
    nv::Image32sC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 32);

    cpu_src1 >> npp_src1;

    cpu_src1.RShift(constVal, cpu_dst);
    npp_src1.RShift(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.RShift(constVal);
    npp_dst.RShift(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC4A", "[NPP.Arithmetic.RShift]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC4::GetStreamContext();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> npp_res(size, size);
    nv::Image32sC4 npp_src1(size, size);
    nv::Image32sC4 npp_dst(size, size);
    cpu::ImageView<Pixel32sC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel32sC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127272727});
    npp_dst.Set({127272727}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 temp;
    op(temp);
    uint constVal = to_uint(temp.x % 32);

    cpu_src1 >> npp_src1;

    cpu_src1A.RShift(constVal, cpu_dstA);
    npp_src1.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.RShift(constVal, cpu_dstA);
    npp_dst.RShiftA(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}