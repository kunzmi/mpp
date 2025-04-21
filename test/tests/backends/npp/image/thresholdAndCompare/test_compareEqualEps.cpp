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

TEST_CASE("32fC1", "[NPP.Statistics.CompareEqualEps]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_src2(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    cpu_src1(0, 0) = 0;
    cpu_src1(1, 1) = 0;
    cpu_src2(0, 0) = 0.05f;
    cpu_src2(1, 1) = -0.05f;
    cpu_src2(1, 0) = cpu_src1(1, 0) + 0.051f;
    cpu_src2(2, 1) = cpu_src1(2, 1) - 0.051f;
    cpu_src2(3, 0) = cpu_src1(3, 0) + 0.049f;
    cpu_src2(4, 1) = cpu_src1(4, 1) - 0.049f;

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.CompareEqualEps(npp_src2, npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CompareEqEps(cpu_src2, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
    CHECK(cpu_dst(0, 0) == 255);
    CHECK(cpu_dst(1, 1) == 255);
    CHECK(cpu_dst(1, 0) == 0);
    CHECK(cpu_dst(2, 1) == 0);
    CHECK(cpu_dst(3, 0) == 255);
    CHECK(cpu_dst(4, 1) == 255);
}

TEST_CASE("32fC1", "[NPP.Statistics.CompareEqualEpsC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel32fC1 cpu_src2;
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    opp::FillRandom<Pixel32fC1> opRand(seed + 1);
    cpu_src1.FillRandom(seed);
    opRand(cpu_src2);

    cpu_src1 >> npp_src1;

    npp_src1.CompareEqualEps(cpu_src2, npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CompareEqEps(cpu_src2, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Statistics.CompareEqualEps]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_src2(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.CompareEqualEps(npp_src2, npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CompareEqEps(cpu_src2, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Statistics.CompareEqualEpsC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel32fC3 cpu_src2;
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    opp::FillRandom<Pixel32fC3> opRand(seed + 1);
    cpu_src1.FillRandom(seed);
    opRand(cpu_src2);

    cpu_src1 >> npp_src1;

    npp_src1.CompareEqualEps(cpu_src2, npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CompareEqEps(cpu_src2, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Statistics.CompareEqualEps]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_src2(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.CompareEqualEps(npp_src2, npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CompareEqEps(cpu_src2, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Statistics.CompareEqualEpsC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    Pixel32fC4 cpu_src2;
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    opp::FillRandom<Pixel32fC4> opRand(seed + 1);
    cpu_src1.FillRandom(seed);
    opRand(cpu_src2);

    cpu_src1 >> npp_src1;

    npp_src1.CompareEqualEps(cpu_src2, npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CompareEqEps(cpu_src2, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Statistics.CompareEqualEps]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A(cpu_src1);
    cpu::ImageView<Pixel32fC4A> cpu_src2A(cpu_src2);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_src2(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.CompareEqualEpsA(npp_src2, npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.CompareEqEps(cpu_src2A, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Statistics.CompareEqualEpsC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A(cpu_src1);
    Pixel32fC4A cpu_src2;
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    opp::FillRandom<Pixel32fC4A> opRand(seed + 1);
    cpu_src1.FillRandom(seed);
    opRand(cpu_src2);

    cpu_src1 >> npp_src1;

    npp_src1.CompareEqualEpsA(cpu_src2.XYZ(), npp_dst, 0.05f, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.CompareEqEps(cpu_src2, 0.05f, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));
}
