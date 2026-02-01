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
#include <common/colorConversion/colorMatrices.h>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

TEST_CASE("8uC3", "[NPP.ColorConversion.GradientColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(256, 256);
    cpu::Image<Pixel8uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormL1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::L1);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormL2, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::L2);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormInf, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::Inf);

    CHECK(npp_res1.IsIdentical(cpu_dst1));
}

TEST_CASE("16uC3", "[NPP.ColorConversion.GradientColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(256, 256);
    cpu::Image<Pixel16uC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel16uC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image16uC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image16uC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormL1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::L1);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormL2, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::L2);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormInf, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::Inf);

    CHECK(npp_res1.IsIdentical(cpu_dst1));
}

TEST_CASE("32fC3", "[NPP.ColorConversion.GradientColorToGray]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(256, 256);
    cpu::Image<Pixel32fC1> cpu_dst1(cpu_src1.SizeAlloc());
    cpu::Image<Pixel32fC1> npp_res1(cpu_src1.SizeAlloc());
    nv::Image32fC3 npp_src1(cpu_src1.SizeAlloc());
    nv::Image32fC1 npp_dst1(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(seed);
    cpu_src1 >> npp_src1;

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormL1, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::L1);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 2));

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormL2, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::L2);

    CHECK(npp_res1.IsSimilar(cpu_dst1, 1));

    npp_src1.GradientColorToGray(npp_dst1, NppiNorm::nppiNormInf, nppCtx);
    npp_res1 << npp_dst1;

    cpu_src1.GradientColorToGray(cpu_dst1, Norm::Inf);

    CHECK(npp_res1.IsIdentical(cpu_dst1));
}
