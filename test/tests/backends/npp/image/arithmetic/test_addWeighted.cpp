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
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image32fcC1View.h>
#include <backends/npp/image/image32fcC2View.h>
#include <backends/npp/image/image32fcC3View.h>
#include <backends/npp/image/image32fcC4View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sc.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
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

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Arithmetic.AddWeighted]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    npp_dst.AddWeighted(npp_src1, 0.3f, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("8uC1", "[NPP.Arithmetic.AddWeightedMasked]")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed                  = Catch::getSeed();
    NppStreamContext nppCtx          = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    npp_dst.AddWeighted(npp_src1, npp_mask, 0.3f, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.AddWeighted]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    npp_dst.AddWeighted(npp_src1, 0.3f, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.002f));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.AddWeightedMasked]")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed                  = Catch::getSeed();
    NppStreamContext nppCtx          = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_mask(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    npp_dst.AddWeighted(npp_src1, npp_mask, 0.3f, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.002f));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.AddWeighted]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    cpu_src1.AddWeighted(cpu_dst, 0.3f);
    npp_dst.AddWeighted(npp_src1, 0.3f, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.AddWeightedMasked]")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed                  = Catch::getSeed();
    NppStreamContext nppCtx          = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_mask(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.FillRandom(seed + 1);

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    cpu_src1.AddWeightedMasked(cpu_dst, 0.3f, cpu_mask);
    npp_dst.AddWeighted(npp_src1, npp_mask, 0.3f, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}
