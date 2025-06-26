#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image64fC1View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/safeCast.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size       = 256;
constexpr int filterSize = 11;

TEST_CASE("8uC1", "[NPP.Statistics.RectStdDev]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dstSum(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dstSqr(size + 1, size + 1);
    cpu::Image<Pixel32fC1> cpu_dstStd(size + 1, size + 1);
    cpu::Image<Pixel32fC1> npp_resSum(size + 1, size + 1);
    cpu::Image<Pixel64fC1> npp_resSqr(size + 1, size + 1);
    cpu::Image<Pixel32fC1> npp_resStd(size + 1, size + 1);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dstSum(size + 1, size + 1);
    nv::Image64fC1 npp_dstSqr(size + 1, size + 1);
    nv::Image32fC1 npp_dstStd(size + 1, size + 1);

    cpu_src1.FillRandom(seed);
    npp_dstStd.Set(0, nppCtx);
    cpu_dstStd.Set(0);

    cpu_src1 >> npp_src1;
    npp_src1.SqrIntegral(npp_dstSum, npp_dstSqr, 0, 0, nppCtx);
    npp_resSum << npp_dstSum;
    npp_resSqr << npp_dstSqr;

    cpu_src1.SqrIntegral(cpu_dstSum, cpu_dstSqr, 0, 0);

    CHECK(cpu_dstSum.IsIdentical(npp_resSum));
    CHECK(cpu_dstSqr.IsIdentical(npp_resSqr));

    npp_dstSum.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    npp_dstSqr.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    npp_dstStd.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    npp_dstSum.RectStdDev(npp_dstSqr, npp_dstStd, NppiRect{0, 0, filterSize, filterSize}, nppCtx);
    npp_dstSum.ResetRoi();
    npp_dstSqr.ResetRoi();
    npp_dstStd.ResetRoi();
    npp_resStd << npp_dstStd;

    cpu_dstStd.SetRoi(Roi(0, 0, size - filterSize, size - filterSize));
    cpu_dstSum.RectStdDev(cpu_dstSqr, cpu_dstStd, FilterArea(filterSize, 0));
    cpu_dstSum.ResetRoi();
    cpu_dstSqr.ResetRoi();
    cpu_dstStd.ResetRoi();

    CHECK(cpu_dstStd.IsIdentical(npp_resStd));
}