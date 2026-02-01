#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC2View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32u.h>
#include <backends/npp/image/image32uC1View.h>
#include <backends/npp/image/image32uC2View.h>
#include <backends/npp/image/image32uC3View.h>
#include <backends/npp/image/image32uC4View.h>
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

TEST_CASE("8uC3", "[NPP.ColorConversion.CFA.GRBG]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> bird = cpu::Image<Pixel8uC3>::Load(root / "bird.tif");
    cpu::Image<Pixel8uC1> cpu_src1(bird.SizeAlloc());
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    bird.RGBToCFA(cpu_src1, BayerGridPosition::GRBG);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGB(npp_dst, NppiBayerGridPosition::NPPI_BAYER_GRBG, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                      nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, BayerGridPosition::GRBG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.CFA.BGGR]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGB(npp_dst, NppiBayerGridPosition::NPPI_BAYER_BGGR, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                      nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, BayerGridPosition::BGGR);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.CFA.GBRG]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGB(npp_dst, NppiBayerGridPosition::NPPI_BAYER_GBRG, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                      nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, BayerGridPosition::GBRG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.CFA.RGGB]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGB(npp_dst, NppiBayerGridPosition::NPPI_BAYER_RGGB, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                      nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, BayerGridPosition::RGGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC4A", "[NPP.ColorConversion.CFA.GRBG]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC4> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGBA(npp_dst, NppiBayerGridPosition::NPPI_BAYER_GRBG, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                       255, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, 255, BayerGridPosition::GRBG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC4A", "[NPP.ColorConversion.CFA.BGGR]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC4> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGBA(npp_dst, NppiBayerGridPosition::NPPI_BAYER_BGGR, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                       255, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, 255, BayerGridPosition::BGGR);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC4A", "[NPP.ColorConversion.CFA.GBRG]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC4> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGBA(npp_dst, NppiBayerGridPosition::NPPI_BAYER_GBRG, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                       255, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, 255, BayerGridPosition::GBRG);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("8uC4A", "[NPP.ColorConversion.CFA.RGGB]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(256, 256);
    cpu::Image<Pixel8uC4> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel8uC4> npp_res(cpu_src1.SizeAlloc());
    nv::Image8uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image8uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGBA(npp_dst, NppiBayerGridPosition::NPPI_BAYER_RGGB, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                       255, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, 255, BayerGridPosition::RGGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("16uC3", "[NPP.ColorConversion.CFA.RGGB]")
{
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(256, 256);
    cpu::Image<Pixel16uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel16uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image16uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image16uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGB(npp_dst, NppiBayerGridPosition::NPPI_BAYER_RGGB, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                      nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, BayerGridPosition::RGGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("16uC4A", "[NPP.ColorConversion.CFA.RGGB]")
{
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(256, 256);
    cpu::Image<Pixel16uC4> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel16uC4> npp_res(cpu_src1.SizeAlloc());
    nv::Image16uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image16uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGBA(npp_dst, NppiBayerGridPosition::NPPI_BAYER_RGGB, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                       255, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, 255, BayerGridPosition::RGGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("32uC3", "[NPP.ColorConversion.CFA.RGGB]")
{
    NppStreamContext nppCtx = nv::Image32uC1::GetStreamContext();

    cpu::Image<Pixel32uC1> cpu_src1(256, 256);
    cpu::Image<Pixel32uC3> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel32uC3> npp_res(cpu_src1.SizeAlloc());
    nv::Image32uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image32uC3 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGB(npp_dst, NppiBayerGridPosition::NPPI_BAYER_RGGB, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                      nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, BayerGridPosition::RGGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}

TEST_CASE("32uC4A", "[NPP.ColorConversion.CFA.RGGB]")
{
    NppStreamContext nppCtx = nv::Image32uC1::GetStreamContext();

    cpu::Image<Pixel32uC1> cpu_src1(256, 256);
    cpu::Image<Pixel32uC4> cpu_dst(cpu_src1.SizeAlloc());
    cpu::Image<Pixel32uC4> npp_res(cpu_src1.SizeAlloc());
    nv::Image32uC1 npp_src1(cpu_src1.SizeAlloc());
    nv::Image32uC4 npp_dst(cpu_src1.SizeAlloc());

    cpu_src1.FillRandom(0);
    cpu_src1.Div(8);

    cpu_src1 >> npp_src1;

    npp_src1.CFAToRGBA(npp_dst, NppiBayerGridPosition::NPPI_BAYER_RGGB, NppiInterpolationMode::NPPI_INTER_UNDEFINED,
                       255, nppCtx);
    npp_res << npp_dst;

    cpu_src1.CFAToRGB(cpu_dst, 255, BayerGridPosition::RGGB);

    CHECK(npp_res.IsSimilar(cpu_dst, 1));
}
