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
#include <common/defines.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.HistogramEven]")
{
    // Note: NPP claims to use the EvenLevels function internally to create evenely spaced bins, but obviously this is
    // not the case. Even for a histogram size of 16 bins for 256 unsigned char values, HistogramEven doesn't create an
    // even distribution, the last bin is way too small... so compare our implementation to Histogram range instead
    // using a distribution created by EvenLevels(). But this is also only compatible for some specific cases of
    // historgams bin counts...

    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    nv::Image8uC1 npp_src1(size, size);

    constexpr int histSize = 16;
    constexpr int nLevels  = histSize + 1;
    opp::cuda::DevVar<int> npp_dst(histSize);
    opp::cuda::DevVar<int> npp_levels(nLevels);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.HistogramRangeGetBufferSize(nLevels, nppCtx));

    std::vector<int> cpu_levels = npp_src1.EvenLevels(nLevels, 0, 256);
    std::vector<int> cpu_dst(histSize);
    std::vector<int> npp_res(histSize);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    npp_levels << cpu_levels;

    npp_src1.HistogramRange(npp_dst, npp_levels, nLevels, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.HistogramEven(reinterpret_cast<Pixel32sC1 *>(cpu_dst.data()), histSize, 0, 256);

    int sumCpu = 0;
    int sumNpp = 0;
    for (size_t i = 0; i < histSize; i++)
    {
        sumCpu += cpu_dst[i];
        sumNpp += npp_res[i];
        CHECK(cpu_dst[i] == npp_res[i]);
    }
}

TEST_CASE("32fC3", "[NPP.Statistics.HistogramEven]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    nv::Image32fC3 npp_src1(size, size);

    constexpr int histSize = 16;

    std::vector<Pixel32sC3> cpu_dst(histSize);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.HistogramEven(cpu_dst.data(), histSize, 0, 1.00001f);

    Pixel32sC3 sumCpu = 0;
    for (size_t i = 0; i < histSize; i++)
    {
        sumCpu += cpu_dst[i];
    }

    CHECK(sumCpu.x == size * size);
    CHECK(sumCpu.y == size * size);
    CHECK(sumCpu.z == size * size);
}