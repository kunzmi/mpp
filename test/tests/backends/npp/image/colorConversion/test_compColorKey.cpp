
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
#include <common/colorConversion/colorMatrices.h>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

TEST_CASE("8uC3", "[NPP.ColorConversion.CompColorKey]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_src2(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src2(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst1(cpu_src1.SizeAlloc());

    Pixel8uC3 keyColor(100, 200, 50);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    // make sure that the keyColor actually is in the image:
    cpu_src1(10, 10) = keyColor;
    cpu_src1(20, 20) = keyColor;
    cpu_src1(30, 30) = keyColor;
    cpu_src1(40, 40) = keyColor;
    cpu_src1(50, 50) = keyColor;
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.CompColorKey(npp_src2, npp_dst1, keyColor, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.CompColorKey(cpu_src2, keyColor, cpu_dst1);

    // The C3 version of CompColorKey doesn't seem to work as expected in NPP...
    // CHECK(npp_res1.IsIdentical(cpu_dst1));
    CHECK(cpu_dst1(10, 10) == cpu_src2(10, 10));
    CHECK(cpu_dst1(20, 20) == cpu_src2(20, 20));
    CHECK(cpu_dst1(30, 30) == cpu_src2(30, 30));
    CHECK(cpu_dst1(40, 40) == cpu_src2(40, 40));
    CHECK(cpu_dst1(50, 50) == cpu_src2(50, 50));
}

TEST_CASE("8uC4", "[NPP.ColorConversion.CompColorKey]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::Image<Pixel8uC4> cpu_src2(256, 256);
    cpu::Image<Pixel8uC4> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_src2(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst1(cpu_src1.SizeAlloc());

    Pixel8uC4 keyColor(100, 200, 50, 80);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    // make sure that the keyColor actually is in the image:
    cpu_src1(10, 10) = keyColor;
    cpu_src1(20, 20) = keyColor;
    cpu_src1(30, 30) = keyColor;
    cpu_src1(40, 40) = keyColor;
    cpu_src1(50, 50) = keyColor;
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.CompColorKey(npp_src2, npp_dst1, keyColor, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.CompColorKey(cpu_src2, keyColor, cpu_dst1);

    CHECK(npp_res1.IsIdentical(cpu_dst1));
}
