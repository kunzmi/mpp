#include <backends/npp/image/image16f.h>
#include <backends/npp/image/image16fC1View.h>
#include <backends/npp/image/image16fC2View.h>
#include <backends/npp/image/image16fC3View.h>
#include <backends/npp/image/image16fC4View.h>
#include <backends/npp/image/image16sc.h>
#include <backends/npp/image/image16scC1View.h>
#include <backends/npp/image/image16scC2View.h>
#include <backends/npp/image/image16scC3View.h>
#include <backends/npp/image/image16scC4View.h>
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
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fcC1View.h>
#include <backends/npp/image/image32fcC2View.h>
#include <backends/npp/image/image32fcC3View.h>
#include <backends/npp/image/image32fcC4View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32sc.h>
#include <backends/npp/image/image32scC1View.h>
#include <backends/npp/image/image32scC2View.h>
#include <backends/npp/image/image32scC3View.h>
#include <backends/npp/image/image32scC4View.h>
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
#include <filesystem>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Arithmetic.AddProduct]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_dst >> npp_dst;

    cpu_src1.AddProduct(cpu_src2, cpu_dst);
    npp_dst.AddProduct(npp_src1, npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("8uC1", "[NPP.Arithmetic.AddProductMasked]")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed                  = Catch::getSeed();
    NppStreamContext nppCtx          = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_dst >> npp_dst;

    cpu_src1.AddProductMasked(cpu_src2, cpu_dst, cpu_mask);
    npp_dst.AddProduct(npp_src1, npp_src2, npp_mask, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.AddProduct]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed);
    // Both, NPP and OPP compile the multiplication + add to a single FMA instruction, which behaves with different
    // rounding behavior than the CPU code for values larger than exact integers as floats. This is known especially for
    // CPU-Debug code. To avoid conflicts with large values, we restrict the test to smaller values:
    cpu_src1.Div(256);
    cpu_src2.Div(256);
    cpu_dst.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_dst >> npp_dst;

    cpu_src1.AddProduct(cpu_src2, cpu_dst);
    npp_dst.AddProduct(npp_src1, npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.001f));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.AddProductMasked]")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed                  = Catch::getSeed();
    NppStreamContext nppCtx          = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_mask(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed);
    // Both, NPP and OPP compile the multiplication + add to a single FMA instruction, which behaves with different
    // rounding behavior than the CPU code for values larger than exact integers as floats. This is known especially for
    // CPU-Debug code. To avoid conflicts with large values, we restrict the test to smaller values:
    cpu_src1.Div(256);
    cpu_src2.Div(256);
    cpu_dst.FillRandom(seed + 1);

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_dst >> npp_dst;

    cpu_src1.AddProductMasked(cpu_src2, cpu_dst, cpu_mask);
    npp_dst.AddProduct(npp_src1, npp_src2, npp_mask, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.001f));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.AddProduct]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_dst >> npp_dst;

    cpu_src1.AddProduct(cpu_src2, cpu_dst);
    npp_dst.AddProduct(npp_src1, npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.AddProductMasked]")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed                  = Catch::getSeed();
    NppStreamContext nppCtx          = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_mask(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_dst >> npp_dst;

    cpu_src1.AddProductMasked(cpu_src2, cpu_dst, cpu_mask);
    npp_dst.AddProduct(npp_src1, npp_src2, npp_mask, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.AddProduct]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_src2(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_src2(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;
    cpu_dst >> npp_dst;

    cpu_src1.AddProduct(cpu_src2, cpu_dst);
    npp_dst.AddProduct(npp_src1, npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.001_hf));
}