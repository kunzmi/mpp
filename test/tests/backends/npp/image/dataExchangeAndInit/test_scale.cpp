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

constexpr int size = 128;

TEST_CASE("8uC1", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    nv::Image8uC1 npp_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst32f(size, size);
    cpu::Image<Pixel32fC1> npp_res32f(size, size);
    nv::Image32fC1 npp_dst32f(size, size);
    cpu::Image<Pixel16sC1> cpu_dst16s(size, size);
    cpu::Image<Pixel16sC1> npp_res16s(size, size);
    nv::Image16sC1 npp_dst16s(size, size);
    cpu::Image<Pixel16uC1> cpu_dst16u(size, size);
    cpu::Image<Pixel16uC1> npp_res16u(size, size);
    nv::Image16uC1 npp_dst16u(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Scale(npp_dst32f, 2.0f, 200.0f, nppCtx);
    npp_res32f << npp_dst32f;

    cpu_src1.Scale(cpu_dst32f, 2.0f, 200.0f);

    CHECK(cpu_dst32f.IsSimilar(npp_res32f, 0.0001f));

    npp_src1.Scale(npp_dst16u, nppCtx);
    npp_res16u << npp_dst16u;

    cpu_src1.Scale(cpu_dst16u, RoundingMode::TowardZero);

    CHECK(cpu_dst16u.IsIdentical(npp_res16u));

    npp_src1.Scale(npp_dst16s, nppCtx);
    npp_res16s << npp_dst16s;

    cpu_src1.Scale(cpu_dst16s, RoundingMode::TowardZero);

    CHECK(cpu_dst16s.IsIdentical(npp_res16s));
}

TEST_CASE("16sC1", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    nv::Image16sC1 npp_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC1> npp_res8u(size, size);
    nv::Image8uC1 npp_dst8u(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_src1.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("16uC1", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    nv::Image16uC1 npp_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC1> npp_res8u(size, size);
    nv::Image8uC1 npp_dst8u(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_src1.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("32sC1", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    nv::Image32sC1 npp_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC1> npp_res8u(size, size);
    nv::Image8uC1 npp_dst8u(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_src1.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("32fC1", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    nv::Image32fC1 npp_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC1> npp_res8u(size, size);
    nv::Image8uC1 npp_dst8u(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Scale(npp_dst8u, 0.0f, 1.0f, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_src1.Scale(cpu_dst8u, 0.0f, 1.0f, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("8uC3", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_srC3(size, size);
    nv::Image8uC3 npp_srC3(size, size);
    cpu::Image<Pixel32fC3> cpu_dst32f(size, size);
    cpu::Image<Pixel32fC3> npp_res32f(size, size);
    nv::Image32fC3 npp_dst32f(size, size);
    cpu::Image<Pixel16sC3> cpu_dst16s(size, size);
    cpu::Image<Pixel16sC3> npp_res16s(size, size);
    nv::Image16sC3 npp_dst16s(size, size);
    cpu::Image<Pixel16uC3> cpu_dst16u(size, size);
    cpu::Image<Pixel16uC3> npp_res16u(size, size);
    nv::Image16uC3 npp_dst16u(size, size);

    cpu_srC3.FillRandom(seed);

    cpu_srC3 >> npp_srC3;

    npp_srC3.Scale(npp_dst32f, 2.0f, 200.0f, nppCtx);
    npp_res32f << npp_dst32f;

    cpu_srC3.Scale(cpu_dst32f, 2.0f, 200.0f);

    CHECK(cpu_dst32f.IsSimilar(npp_res32f, 0.0001f));

    npp_srC3.Scale(npp_dst16u, nppCtx);
    npp_res16u << npp_dst16u;

    cpu_srC3.Scale(cpu_dst16u, RoundingMode::TowardZero);

    CHECK(cpu_dst16u.IsIdentical(npp_res16u));

    npp_srC3.Scale(npp_dst16s, nppCtx);
    npp_res16s << npp_dst16s;

    cpu_srC3.Scale(cpu_dst16s, RoundingMode::TowardZero);

    CHECK(cpu_dst16s.IsIdentical(npp_res16s));
}

TEST_CASE("16sC3", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_srC3(size, size);
    nv::Image16sC3 npp_srC3(size, size);
    cpu::Image<Pixel8uC3> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC3> npp_res8u(size, size);
    nv::Image8uC3 npp_dst8u(size, size);

    cpu_srC3.FillRandom(seed);

    cpu_srC3 >> npp_srC3;

    npp_srC3.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC3.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("16uC3", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_srC3(size, size);
    nv::Image16uC3 npp_srC3(size, size);
    cpu::Image<Pixel8uC3> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC3> npp_res8u(size, size);
    nv::Image8uC3 npp_dst8u(size, size);

    cpu_srC3.FillRandom(seed);

    cpu_srC3 >> npp_srC3;

    npp_srC3.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC3.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("32sC3", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC3::GetStreamContext();

    cpu::Image<Pixel32sC3> cpu_srC3(size, size);
    nv::Image32sC3 npp_srC3(size, size);
    cpu::Image<Pixel8uC3> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC3> npp_res8u(size, size);
    nv::Image8uC3 npp_dst8u(size, size);

    cpu_srC3.FillRandom(seed);

    cpu_srC3 >> npp_srC3;

    npp_srC3.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC3.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("32fC3", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_srC3(size, size);
    nv::Image32fC3 npp_srC3(size, size);
    cpu::Image<Pixel8uC3> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC3> npp_res8u(size, size);
    nv::Image8uC3 npp_dst8u(size, size);

    cpu_srC3.FillRandom(seed);

    cpu_srC3 >> npp_srC3;

    npp_srC3.Scale(npp_dst8u, 0.0f, 1.0f, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC3.Scale(cpu_dst8u, 0.0f, 1.0f, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("8uC4", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_srC4(size, size);
    nv::Image8uC4 npp_srC4(size, size);
    cpu::Image<Pixel32fC4> cpu_dst32f(size, size);
    cpu::Image<Pixel32fC4> npp_res32f(size, size);
    nv::Image32fC4 npp_dst32f(size, size);
    cpu::Image<Pixel16sC4> cpu_dst16s(size, size);
    cpu::Image<Pixel16sC4> npp_res16s(size, size);
    nv::Image16sC4 npp_dst16s(size, size);
    cpu::Image<Pixel16uC4> cpu_dst16u(size, size);
    cpu::Image<Pixel16uC4> npp_res16u(size, size);
    nv::Image16uC4 npp_dst16u(size, size);

    cpu_srC4.FillRandom(seed);

    cpu_srC4 >> npp_srC4;

    npp_srC4.Scale(npp_dst32f, 2.0f, 200.0f, nppCtx);
    npp_res32f << npp_dst32f;

    cpu_srC4.Scale(cpu_dst32f, 2.0f, 200.0f);

    CHECK(cpu_dst32f.IsSimilar(npp_res32f, 0.0001f));

    npp_srC4.Scale(npp_dst16u, nppCtx);
    npp_res16u << npp_dst16u;

    cpu_srC4.Scale(cpu_dst16u, RoundingMode::TowardZero);

    CHECK(cpu_dst16u.IsIdentical(npp_res16u));

    npp_srC4.Scale(npp_dst16s, nppCtx);
    npp_res16s << npp_dst16s;

    cpu_srC4.Scale(cpu_dst16s, RoundingMode::TowardZero);

    CHECK(cpu_dst16s.IsIdentical(npp_res16s));
}

TEST_CASE("16sC4", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_srC4(size, size);
    nv::Image16sC4 npp_srC4(size, size);
    cpu::Image<Pixel8uC4> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC4> npp_res8u(size, size);
    nv::Image8uC4 npp_dst8u(size, size);

    cpu_srC4.FillRandom(seed);

    cpu_srC4 >> npp_srC4;

    npp_srC4.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC4.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("16uC4", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_srC4(size, size);
    nv::Image16uC4 npp_srC4(size, size);
    cpu::Image<Pixel8uC4> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC4> npp_res8u(size, size);
    nv::Image8uC4 npp_dst8u(size, size);

    cpu_srC4.FillRandom(seed);

    cpu_srC4 >> npp_srC4;

    npp_srC4.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC4.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("32sC4", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC4::GetStreamContext();

    cpu::Image<Pixel32sC4> cpu_srC4(size, size);
    nv::Image32sC4 npp_srC4(size, size);
    cpu::Image<Pixel8uC4> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC4> npp_res8u(size, size);
    nv::Image8uC4 npp_dst8u(size, size);

    cpu_srC4.FillRandom(seed);

    cpu_srC4 >> npp_srC4;

    npp_srC4.Scale(npp_dst8u, NppHintAlgorithm::NPP_ALG_HINT_NONE, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC4.Scale(cpu_dst8u, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}

TEST_CASE("32fC4", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_srC4(size, size);
    nv::Image32fC4 npp_srC4(size, size);
    cpu::Image<Pixel8uC4> cpu_dst8u(size, size);
    cpu::Image<Pixel8uC4> npp_res8u(size, size);
    nv::Image8uC4 npp_dst8u(size, size);

    cpu_srC4.FillRandom(seed);

    cpu_srC4 >> npp_srC4;

    npp_srC4.Scale(npp_dst8u, 0.0f, 1.0f, nppCtx);
    npp_res8u << npp_dst8u;

    cpu_srC4.Scale(cpu_dst8u, 0.0f, 1.0f, RoundingMode::TowardZero);

    CHECK(cpu_dst8u.IsIdentical(npp_res8u));
}