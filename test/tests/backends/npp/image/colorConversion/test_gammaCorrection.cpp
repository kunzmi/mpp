
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

TEST_CASE("8uC3", "[NPP.ColorConversion.GammaFwd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.GammaFwd(npp_dst1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GammaCorrBT709(cpu_dst1, 255.0f);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
}

TEST_CASE("8uC4A", "[NPP.ColorConversion.GammaFwd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC4A> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res1(cpu_src1.SizeAlloc());
    cpu::ImageView<Pixel8uC4A> npp_res1A(npp_res1);
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.GammaFwdA(npp_dst1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1A.GammaCorrBT709(cpu_dst1, 255.0f);

    CHECK(npp_res1A.IsSimilar(cpu_dst1, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.GammaFwd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.GammaInv(npp_dst1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GammaInvCorrBT709(cpu_dst1, 255.0f);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));
}

TEST_CASE("8uC4A", "[NPP.ColorConversion.GammaFwd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(256, 256);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC4A> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res1(cpu_src1.SizeAlloc());
    cpu::ImageView<Pixel8uC4A> npp_res1A(npp_res1);
    nv::Image8uC4 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.GammaInvA(npp_dst1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1A.GammaInvCorrBT709(cpu_dst1, 255.0f);

    CHECK(npp_res1A.IsSimilar(cpu_dst1, 1));
}
