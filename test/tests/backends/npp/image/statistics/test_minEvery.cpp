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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_srcdst(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_srcdst(size, size);
    nv::Image8uC1 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_srcdst(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_srcdst(size, size);
    nv::Image8uC3 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_srcdst(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_srcdst(size, size);
    nv::Image8uC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_srcdst(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_srcdstA(cpu_srcdst);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src2A(cpu_src2);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_srcdst(size, size);
    nv::Image8uC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEveryA(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdstA.MinEvery(cpu_src2A);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_srcdst(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_srcdst(size, size);
    nv::Image16uC1 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_srcdst(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_srcdst(size, size);
    nv::Image16uC3 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_srcdst(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_srcdst(size, size);
    nv::Image16uC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_srcdst(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_srcdstA(cpu_srcdst);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src2A(cpu_src2);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_srcdst(size, size);
    nv::Image16uC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEveryA(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdstA.MinEvery(cpu_src2A);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16sC1", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_srcdst(size, size);
    cpu::Image<Pixel16sC1> cpu_src2(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image16sC1 npp_srcdst(size, size);
    nv::Image16sC1 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16sC3", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_srcdst(size, size);
    cpu::Image<Pixel16sC3> cpu_src2(size, size);
    cpu::Image<Pixel16sC3> npp_res(size, size);
    nv::Image16sC3 npp_srcdst(size, size);
    nv::Image16sC3 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16sC4", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_srcdst(size, size);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_srcdst(size, size);
    nv::Image16sC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("16sC4A", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_srcdst(size, size);
    cpu::ImageView<Pixel16sC4A> cpu_srcdstA(cpu_srcdst);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    cpu::ImageView<Pixel16sC4A> cpu_src2A(cpu_src2);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_srcdst(size, size);
    nv::Image16sC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEveryA(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdstA.MinEvery(cpu_src2A);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_srcdst(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_srcdst(size, size);
    nv::Image32fC1 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_srcdst(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_srcdst(size, size);
    nv::Image32fC3 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_srcdst(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_srcdst(size, size);
    nv::Image32fC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEvery(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdst.MinEvery(cpu_src2);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Statistics.MinEvery]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_srcdst(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_srcdstA(cpu_srcdst);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src2A(cpu_src2);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_srcdst(size, size);
    nv::Image32fC4 npp_src2(size, size);

    cpu_srcdst.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_srcdst >> npp_srcdst;
    cpu_src2 >> npp_src2;

    npp_srcdst.MinEveryA(npp_src2, nppCtx);
    npp_res << npp_srcdst;

    cpu_srcdstA.MinEvery(cpu_src2A);

    CHECK(cpu_srcdst.IsIdentical(npp_res));
}