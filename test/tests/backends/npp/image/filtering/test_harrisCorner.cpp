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

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Filtering.HarrisCorner]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load(root / "bird256bw.tif");
    cpu::Image<Pixel32fC4> cpu_covar(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::ImageView<Pixel16sC1> cpu_null(nullptr, {{0, 0}, 0});
    cpu::ImageView<Pixel32fC1> cpu_null2(nullptr, {{0, 0}, 0});
    cpu::Image<Pixel16sC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAng(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);
    opp::cuda::DevVar<byte> buffer(npp_src1.FilterHarrisCornersBorderGetBufferSize());

    cpu_src1 >> npp_src1;

    npp_src1.FilterHarrisCornersBorder(npp_dst, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, 0.04f,
                                       0.00001f, NPP_BORDER_REPLICATE, buffer, nppCtx);
    npp_res << npp_dst;

    cpu_src1.GradientVectorSobel(cpu_null, cpu_null, cpu_null, cpu_null2, cpu_covar, Norm::L2, MaskSize::Mask_3x3,
                                 BorderType::Replicate);

    cpu_covar.HarrisCornerResponse(cpu_dst, 5, 0.04f, 0.00001f, BorderType::Replicate);

    // normalize the result, NPP seems to have some different scaling:
    Pixel32fC1 min;
    Pixel32fC1 max;
    cpu_dst.Min(min);
    cpu_dst.Max(max);
    cpu_dst.Sub(min);
    cpu_dst.Div(max - min);

    npp_res.Min(min);
    npp_res.Max(max);
    npp_res.Sub(min);
    npp_res.Div(max - min);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.000001f));
}