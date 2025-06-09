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

TEST_CASE("32fC1", "[NPP.Morpholgy.MorphGradient]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed            = Catch::getSeed();
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_tpl = cpu::Image<Pixel8uC1>::Load(root / "morph.tif");
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_temp(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_temp1(size, size);
    nv::Image32fC1 npp_temp2(size, size);
    nv::Image32fC1 npp_dst(size, size);
    opp::cuda::DevVar<byte> mask(size_tpl * size_tpl);
    opp::cuda::DevVar<byte> buffer(npp_src1.MorphGetBufferSize());
    mask << cpu_tpl.Pointer();

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.MorphGradientBorder(npp_dst, mask, {size_tpl, size_tpl}, {2, 2}, buffer, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.BlackHat(cpu_temp, cpu_dst, cpu_tpl.Pointer(), size_tpl, BorderType::Replicate);

    // NPP returns "BlackHat" as MorphGradient...
    CHECK(cpu_dst.IsIdentical(npp_res));

    // Computing the "real" MorphGradient
    npp_src1.ErodeBorder(npp_temp1, mask, {size_tpl, size_tpl}, {2, 2}, NPP_BORDER_REPLICATE, nppCtx);
    npp_src1.DilateBorder(npp_temp2, mask, {size_tpl, size_tpl}, {2, 2}, NPP_BORDER_REPLICATE, nppCtx);
    npp_temp1.Sub(npp_temp2, npp_dst, nppCtx);
    npp_res << npp_dst;

    cpu_src1.MorphologyGradient(cpu_dst, cpu_tpl.Pointer(), size_tpl, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}