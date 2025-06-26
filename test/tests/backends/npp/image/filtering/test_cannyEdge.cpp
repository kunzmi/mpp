#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <common/defines.h>
#include <common/safeCast.h>
#include <numbers>
#include <vector>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Filtering.CannyEdge]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_tmp(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::ImageView<Pixel16sC1> cpu_null(nullptr, {{0, 0}, 0});
    cpu::ImageView<Pixel32fC4> cpu_null2(nullptr, {{0, 0}, 0});
    cpu::Image<Pixel16sC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_temp1(size, size);
    cpu::Image<Pixel32fC1> cpu_temp2(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAng(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);
    mpp::cuda::DevVar<byte> buffer(npp_src1.FilterCannyBorderGetBufferSize());

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.FilterCannyBorder(npp_dst, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, 20, 200, nppiNormL1,
                               NPP_BORDER_REPLICATE, buffer, nppCtx);
    npp_res << npp_dst;

    cpu_src1.GradientVectorSobel(cpu_null, cpu_null, cpu_dstMag, cpu_dstAng, cpu_null2, Norm::L1, MaskSize::Mask_3x3,
                                 BorderType::Replicate);
    cpu_dstMag.CannyEdge(cpu_dstAng, cpu_tmp, cpu_dst, 20, 200);

    cpu_dst.Convert(cpu_temp1);
    npp_res.Convert(cpu_temp2);
    cpu_temp1.Sub(cpu_temp2);
    cpu_temp1.Abs();
    Pixel64uC1 errorCount = 0;
    cpu_temp1.CountInRange(200, 300, errorCount);

    // rounding is different and the hysteresis threshold also differs, so some pixels are off:
    CHECK(errorCount < 200);
}