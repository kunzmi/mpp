#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC2View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image8s.h>
#include <backends/npp/image/image8sC1View.h>
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

TEST_CASE("8uC1", "[NPP.Filtering.SobelCrossFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image16sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.FilterSobelCrossBorder(npp_dst, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.FixedFilter(cpu_dst, FixedFilter::SobelCross, MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.FilterSobelCrossBorder(npp_dst, NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.FixedFilter(cpu_dst, FixedFilter::SobelCross, MaskSize::Mask_5x5, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8sC1", "[NPP.Filtering.SobelCrossFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image8sC1 npp_src1(size, size);
    nv::Image16sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.FilterSobelCrossBorder(npp_dst, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.FixedFilter(cpu_dst, FixedFilter::SobelCross, MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.FilterSobelCrossBorder(npp_dst, NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.FixedFilter(cpu_dst, FixedFilter::SobelCross, MaskSize::Mask_5x5, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Filtering.SobelCrossFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.FilterSobelCrossBorder(npp_dst, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.FixedFilter(cpu_dst, FixedFilter::SobelCross, MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    npp_src1.FilterSobelCrossBorder(npp_dst, NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.FixedFilter(cpu_dst, FixedFilter::SobelCross, MaskSize::Mask_5x5, BorderType::Replicate);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}