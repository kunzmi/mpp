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
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/utilities.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Arithmetic.Exp]")
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

    cpu_src1.Exp(cpu_dst);
    npp_src1.Exp(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Exp();
    npp_src1.Exp(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.Exp]")
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

    cpu_src1.Exp(cpu_dst);
    npp_src1.Exp(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Exp();
    npp_src1.Exp(0, nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.Exp]")
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

    cpu_src1.Exp(cpu_dst);
    npp_src1.Exp(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Exp();
    npp_dst.Exp(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.Exp]")
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

    cpu_src1.Exp(cpu_dst);
    npp_src1.Exp(npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Exp();
    npp_dst.Exp(0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.Exp]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src1.Mul(16);
    cpu_src1.Sub(8);

    cpu_src1 >> npp_src1;

    cpu_src1.Exp(cpu_dst);
    npp_src1.Exp(npp_dst, nppCtx);

    npp_res << npp_dst;

    auto iter_cpu = cpu_dst.begin();
    for (auto &npp_pixel : npp_res)
    {
        if (npp_pixel.Value().x == numeric_limits<float>::min() && iter_cpu.Value().x == 0)
        {
            npp_pixel.Value().x = 0;
        }
        if (npp_pixel.Value().x == numeric_limits<float>::max() && isinf(iter_cpu.Value().x))
        {
            iter_cpu.Value().x = numeric_limits<float>::max();
        }
        ++iter_cpu;
    }

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, 0.5f));

    cpu_src1.Exp();
    npp_src1.Exp(nppCtx);

    npp_res << npp_src1;

    auto iter_cpu2 = cpu_src1.begin();
    for (auto &npp_pixel : npp_res)
    {
        if (npp_pixel.Value().x == numeric_limits<float>::min() && iter_cpu2.Value().x == 0)
        {
            npp_pixel.Value().x = 0;
        }
        if (npp_pixel.Value().x == numeric_limits<float>::max() && isinf(iter_cpu2.Value().x))
        {
            iter_cpu2.Value().x = numeric_limits<float>::max();
        }
        ++iter_cpu2;
    }

    CHECK(cpu_src1.IsSimilarIgnoringNAN(npp_res, 0.5f));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.Exp]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src1.Mul(16);
    cpu_src1.Sub(8);

    cpu_src1 >> npp_src1;

    cpu_src1.Exp(cpu_dst);
    npp_src1.Exp(npp_dst, nppCtx);

    npp_res << npp_dst;

    auto iter_cpu = cpu_dst.begin();
    for (auto &npp_pixel : npp_res)
    {
        if (npp_pixel.Value().x == numeric_limits<float>::min() && iter_cpu.Value().x == 0)
        {
            npp_pixel.Value().x = 0;
        }
        if (npp_pixel.Value().x == numeric_limits<float>::max() && isinf(iter_cpu.Value().x))
        {
            iter_cpu.Value().x = numeric_limits<float>::max();
        }
        if (npp_pixel.Value().y == numeric_limits<float>::min() && iter_cpu.Value().y == 0)
        {
            npp_pixel.Value().y = 0;
        }
        if (npp_pixel.Value().y == numeric_limits<float>::max() && isinf(iter_cpu.Value().y))
        {
            iter_cpu.Value().y = numeric_limits<float>::max();
        }
        if (npp_pixel.Value().z == numeric_limits<float>::min() && iter_cpu.Value().z == 0)
        {
            npp_pixel.Value().z = 0;
        }
        if (npp_pixel.Value().z == numeric_limits<float>::max() && isinf(iter_cpu.Value().z))
        {
            iter_cpu.Value().z = numeric_limits<float>::max();
        }
        ++iter_cpu;
    }

    CHECK(cpu_dst.IsSimilarIgnoringNAN(npp_res, 0.5f));

    cpu_src1.Exp();
    npp_src1.Exp(nppCtx);

    npp_res << npp_src1;

    auto iter_cpu2 = cpu_src1.begin();
    for (auto &npp_pixel : npp_res)
    {
        if (npp_pixel.Value().x == numeric_limits<float>::min() && iter_cpu2.Value().x == 0)
        {
            npp_pixel.Value().x = 0;
        }
        if (npp_pixel.Value().x == numeric_limits<float>::max() && isinf(iter_cpu2.Value().x))
        {
            iter_cpu2.Value().x = numeric_limits<float>::max();
        }
        if (npp_pixel.Value().y == numeric_limits<float>::min() && iter_cpu2.Value().y == 0)
        {
            npp_pixel.Value().y = 0;
        }
        if (npp_pixel.Value().y == numeric_limits<float>::max() && isinf(iter_cpu2.Value().y))
        {
            iter_cpu2.Value().y = numeric_limits<float>::max();
        }
        if (npp_pixel.Value().z == numeric_limits<float>::min() && iter_cpu2.Value().z == 0)
        {
            npp_pixel.Value().z = 0;
        }
        if (npp_pixel.Value().z == numeric_limits<float>::max() && isinf(iter_cpu2.Value().z))
        {
            iter_cpu2.Value().z = numeric_limits<float>::max();
        }
        ++iter_cpu2;
    }

    CHECK(cpu_src1.IsSimilarIgnoringNAN(npp_res, 0.5f));
}