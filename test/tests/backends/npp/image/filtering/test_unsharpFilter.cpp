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

std::vector<float> GetFilter(size_t aSize, float aSigma)
{
    std::vector<float> res(aSize);

    float sum_filter = 0;
    for (size_t i = 0; i < aSize; i++)
    {
        float x = to_float(i) - to_float(aSize / 2);
        res[i]  = to_float(1.0f / (std::sqrt(2.0f * std::numbers::pi_v<float>) * aSigma) *
                           std::exp(-(x * x) / (2.0f * aSigma * aSigma)));
        sum_filter += res[i];
    }
    for (size_t i = 0; i < aSize; i++)
    {
        res[i] /= sum_filter;
    }
    return res;
}

TEST_CASE("8uC1", "[NPP.Filtering.UnsharpFilter]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    const Pixel32fC1 radius = 4.0f;
    const Pixel32fC1 sigma  = 1.5f;
    const float weight      = 0.7f;
    const float threshold   = 0.0f;

    cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load(root / "bird256bw.tif");
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);
    mpp::cuda::DevVar<byte> buffer(npp_src1.FilterUnsharpGetBufferSize(radius, sigma));
    std::vector<float> filter = GetFilter(9, sigma.x);

    cpu_src1 >> npp_src1;

    // Something is strange with the threshold parameter in NPP: I get similar results once we divide by 255, but it is
    // not identical neither, with threshold == 0, the results are the same apart from rounding errors...
    npp_src1.FilterUnsharpBorder(npp_dst, radius, sigma, weight, threshold / 255.0f, NPP_BORDER_REPLICATE, buffer,
                                 nppCtx);
    npp_res << npp_dst;

    cpu_src1.UnsharpFilter(cpu_dst, filter.data(), 9, 4, weight, threshold, BorderType::Replicate);

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}
