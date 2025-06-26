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

TEST_CASE("8uC1", "[NPP.Filtering.ThresholdAdaptiveBoxFilter]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    const Pixel32fC1 delta = 4;

    cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load(root / "bird256bw.tif");
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1 >> npp_src1;

    npp_src1.FilterThresholdAdaptiveBoxBorder(npp_dst, {9, 9}, delta, 255, 0, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.ThresholdAdaptiveBoxFilter(cpu_dst, 9, delta, 255, 0, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}
