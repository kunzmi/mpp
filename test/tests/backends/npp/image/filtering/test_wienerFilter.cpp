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

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Filtering.WienerFilter]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1      = cpu::Image<Pixel8uC1>::Load(root / "bird256bwnoisy.tif");
    cpu::Image<Pixel8uC1> cpu_reference = cpu::Image<Pixel8uC1>::Load(root / "bird256bwnoisy_wienerfiltered.tif");
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1 >> npp_src1;

    Pixel32fC1 noise{0.5f};
    Pixel32fC1 noise2{512.0f};
    npp_src1.FilterWienerBorder(npp_dst, {7, 7}, {3, 3}, noise.data(), NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.WienerFilter(cpu_dst, 7, noise2, BorderType::Replicate);

    // I have no idea what the NPP Wiener filter is doing... The resulting image is too dark and why is the noise
    // parameter limited to 0..1? So just compare with a precomputed reference image

    CHECK(cpu_dst.IsIdentical(cpu_reference));
}
