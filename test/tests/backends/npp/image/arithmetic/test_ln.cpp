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
#include <common/half_fp16.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Arithmetic.Ln]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Ln();
    npp_src1.Ln(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.Ln]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Ln();
    npp_src1.Ln(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.Ln]")
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

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Ln();
    npp_dst.Ln(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.Ln]")
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

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Ln();
    npp_dst.Ln(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.Ln]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src1.Mul(1024);
    cpu_src1.Sub(512);

    cpu_src1 >> npp_src1;

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, numeric_limits<HalfFp16>::min()));

    cpu_dst.Ln();
    npp_dst.Ln(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, numeric_limits<HalfFp16>::min()));
}

TEST_CASE("16fC3", "[NPP.Arithmetic.Ln]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC3::GetStreamContext();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> npp_res(size, size);
    nv::Image16fC3 npp_src1(size, size);
    nv::Image16fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src1.Mul(1024);
    cpu_src1.Sub(512);

    cpu_src1 >> npp_src1;

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, numeric_limits<HalfFp16>::min()));

    cpu_dst.Ln();
    npp_dst.Ln(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, numeric_limits<HalfFp16>::min()));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.Ln]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src1.Mul(1024);
    cpu_src1.Sub(512);

    cpu_src1 >> npp_src1;

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, 0.00001f));

    cpu_dst.Ln();
    npp_dst.Ln(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, 0.00001f));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.Ln]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src1.Mul(1024);
    cpu_src1.Sub(512);

    cpu_src1 >> npp_src1;

    cpu_src1.Ln(cpu_dst);
    npp_src1.Ln(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, 0.00001f));

    cpu_dst.Ln();
    npp_dst.Ln(nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, 0.00001f));
}