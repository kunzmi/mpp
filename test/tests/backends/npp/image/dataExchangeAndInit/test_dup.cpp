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
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32u.h>
#include <backends/npp/image/image32uC1View.h>
#include <backends/npp/image/image32uC2View.h>
#include <backends/npp/image/image32uC3View.h>
#include <backends/npp/image/image32uC4View.h>
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image64fC1View.h>
#include <backends/npp/image/image64fC2View.h>
#include <backends/npp/image/image64fC3View.h>
#include <backends/npp/image/image64fC4View.h>
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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.DataExchangeAndInit.Dup]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst3(size, size);
    cpu::Image<Pixel8uC3> npp_res3(size, size);
    cpu::Image<Pixel8uC4> cpu_dst4(size, size);
    cpu::Image<Pixel8uC4> npp_res4(size, size);
    cpu::Image<Pixel8uC4A> cpu_dst4A(size, size);
    cpu::ImageView<Pixel8uC4> cpu_dst4AA(cpu_dst4A);
    cpu::Image<Pixel8uC4> npp_res4A(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC3 npp_dst3(size, size);
    nv::Image8uC4 npp_dst4(size, size);
    nv::Image8uC4 npp_dst4A(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst4AA.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst4AA >> npp_dst4A;

    npp_src1.Dup(npp_dst3, nppCtx);
    npp_src1.Dup(npp_dst4, nppCtx);
    npp_src1.DupA(npp_dst4A, nppCtx);
    npp_res3 << npp_dst3;
    npp_res4 << npp_dst4;
    npp_res4A << npp_dst4A;

    cpu_src1.Dup(cpu_dst3);
    cpu_src1.Dup(cpu_dst4);
    cpu_src1.Dup(cpu_dst4A);

    CHECK(cpu_dst3.IsIdentical(npp_res3));
    CHECK(cpu_dst4.IsIdentical(npp_res4));
    CHECK(cpu_dst4AA.IsIdentical(npp_res4A));
}

TEST_CASE("16sC1", "[NPP.DataExchangeAndInit.Dup]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_dst3(size, size);
    cpu::Image<Pixel16sC3> npp_res3(size, size);
    cpu::Image<Pixel16sC4> cpu_dst4(size, size);
    cpu::Image<Pixel16sC4> npp_res4(size, size);
    cpu::Image<Pixel16sC4A> cpu_dst4A(size, size);
    cpu::ImageView<Pixel16sC4> cpu_dst4AA(cpu_dst4A);
    cpu::Image<Pixel16sC4> npp_res4A(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC3 npp_dst3(size, size);
    nv::Image16sC4 npp_dst4(size, size);
    nv::Image16sC4 npp_dst4A(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst4AA.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst4AA >> npp_dst4A;

    npp_src1.Dup(npp_dst3, nppCtx);
    npp_src1.Dup(npp_dst4, nppCtx);
    npp_src1.DupA(npp_dst4A, nppCtx);
    npp_res3 << npp_dst3;
    npp_res4 << npp_dst4;
    npp_res4A << npp_dst4A;

    cpu_src1.Dup(cpu_dst3);
    cpu_src1.Dup(cpu_dst4);
    cpu_src1.Dup(cpu_dst4A);

    CHECK(cpu_dst3.IsIdentical(npp_res3));
    CHECK(cpu_dst4.IsIdentical(npp_res4));
    CHECK(cpu_dst4AA.IsIdentical(npp_res4A));
}

TEST_CASE("16uC1", "[NPP.DataExchangeAndInit.Dup]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst3(size, size);
    cpu::Image<Pixel16uC3> npp_res3(size, size);
    cpu::Image<Pixel16uC4> cpu_dst4(size, size);
    cpu::Image<Pixel16uC4> npp_res4(size, size);
    cpu::Image<Pixel16uC4A> cpu_dst4A(size, size);
    cpu::ImageView<Pixel16uC4> cpu_dst4AA(cpu_dst4A);
    cpu::Image<Pixel16uC4> npp_res4A(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC3 npp_dst3(size, size);
    nv::Image16uC4 npp_dst4(size, size);
    nv::Image16uC4 npp_dst4A(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst4AA.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst4AA >> npp_dst4A;

    npp_src1.Dup(npp_dst3, nppCtx);
    npp_src1.Dup(npp_dst4, nppCtx);
    npp_src1.DupA(npp_dst4A, nppCtx);
    npp_res3 << npp_dst3;
    npp_res4 << npp_dst4;
    npp_res4A << npp_dst4A;

    cpu_src1.Dup(cpu_dst3);
    cpu_src1.Dup(cpu_dst4);
    cpu_src1.Dup(cpu_dst4A);

    CHECK(cpu_dst3.IsIdentical(npp_res3));
    CHECK(cpu_dst4.IsIdentical(npp_res4));
    CHECK(cpu_dst4AA.IsIdentical(npp_res4A));
}

TEST_CASE("32sC1", "[NPP.DataExchangeAndInit.Dup]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst3(size, size);
    cpu::Image<Pixel32sC3> npp_res3(size, size);
    cpu::Image<Pixel32sC4> cpu_dst4(size, size);
    cpu::Image<Pixel32sC4> npp_res4(size, size);
    cpu::Image<Pixel32sC4A> cpu_dst4A(size, size);
    cpu::ImageView<Pixel32sC4> cpu_dst4AA(cpu_dst4A);
    cpu::Image<Pixel32sC4> npp_res4A(size, size);
    nv::Image32sC1 npp_src1(size, size);
    nv::Image32sC3 npp_dst3(size, size);
    nv::Image32sC4 npp_dst4(size, size);
    nv::Image32sC4 npp_dst4A(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst4AA.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst4AA >> npp_dst4A;

    npp_src1.Dup(npp_dst3, nppCtx);
    npp_src1.Dup(npp_dst4, nppCtx);
    npp_src1.DupA(npp_dst4A, nppCtx);
    npp_res3 << npp_dst3;
    npp_res4 << npp_dst4;
    npp_res4A << npp_dst4A;

    cpu_src1.Dup(cpu_dst3);
    cpu_src1.Dup(cpu_dst4);
    cpu_src1.Dup(cpu_dst4A);

    CHECK(cpu_dst3.IsIdentical(npp_res3));
    CHECK(cpu_dst4.IsIdentical(npp_res4));
    CHECK(cpu_dst4AA.IsIdentical(npp_res4A));
}

TEST_CASE("32fC1", "[NPP.DataExchangeAndInit.Dup]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst3(size, size);
    cpu::Image<Pixel32fC3> npp_res3(size, size);
    cpu::Image<Pixel32fC4> cpu_dst4(size, size);
    cpu::Image<Pixel32fC4> npp_res4(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst4A(size, size);
    cpu::ImageView<Pixel32fC4> cpu_dst4AA(cpu_dst4A);
    cpu::Image<Pixel32fC4> npp_res4A(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC3 npp_dst3(size, size);
    nv::Image32fC4 npp_dst4(size, size);
    nv::Image32fC4 npp_dst4A(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst4AA.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst4AA >> npp_dst4A;

    npp_src1.Dup(npp_dst3, nppCtx);
    npp_src1.Dup(npp_dst4, nppCtx);
    npp_src1.DupA(npp_dst4A, nppCtx);
    npp_res3 << npp_dst3;
    npp_res4 << npp_dst4;
    npp_res4A << npp_dst4A;

    cpu_src1.Dup(cpu_dst3);
    cpu_src1.Dup(cpu_dst4);
    cpu_src1.Dup(cpu_dst4A);

    CHECK(cpu_dst3.IsIdentical(npp_res3));
    CHECK(cpu_dst4.IsIdentical(npp_res4));
    CHECK(cpu_dst4AA.IsIdentical(npp_res4A));
}