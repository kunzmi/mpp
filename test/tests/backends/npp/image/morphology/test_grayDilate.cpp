#include <backends/cuda/devVar.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
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

constexpr int size     = 256;
constexpr int size_tpl = 5;

TEST_CASE("8uC1", "[NPP.Morpholgy.GrayDilate]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed            = Catch::getSeed();
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_tpl = cpu::Image<Pixel8uC1>::Load(root / "morph.tif");
    cpu::Image<Pixel32sC1> cpu_tpl2(size_tpl, size_tpl);
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);
    opp::cuda::DevVar<int> mask(size_tpl * size_tpl);
    cpu_tpl.Convert(cpu_tpl2);
    mask << cpu_tpl2.Pointer();

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GrayDilateBorder(npp_dst, mask, {size_tpl, size_tpl}, {2, 2}, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.DilationGray(cpu_dst, cpu_tpl2.Pointer(), size_tpl, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Morpholgy.GrayDilate]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed            = Catch::getSeed();
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_tpl = cpu::Image<Pixel8uC1>::Load(root / "morph.tif");
    cpu::Image<Pixel32fC1> cpu_tpl2(size_tpl, size_tpl);
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);
    opp::cuda::DevVar<float> mask(size_tpl * size_tpl);
    cpu_tpl.Convert(cpu_tpl2);
    mask << cpu_tpl2.Pointer();

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GrayDilateBorder(npp_dst, mask, {size_tpl, size_tpl}, {2, 2}, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.DilationGray(cpu_dst, cpu_tpl2.Pointer(), size_tpl, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}