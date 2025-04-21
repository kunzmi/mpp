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

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.SqrIntegral32s]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size + 1, size + 1);
    cpu::Image<Pixel32sC1> cpu_dstSqr(size + 1, size + 1);
    cpu::Image<Pixel32sC1> npp_res(size + 1, size + 1);
    cpu::Image<Pixel32sC1> npp_resSqr(size + 1, size + 1);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image32sC1 npp_dst(size + 1, size + 1);
    nv::Image32sC1 npp_dstSqr(size + 1, size + 1);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.SqrIntegral(npp_dst, npp_dstSqr, 12, 144, nppCtx);
    npp_res << npp_dst;
    npp_resSqr << npp_dstSqr;

    cpu_src1.SqrIntegral(cpu_dst, cpu_dstSqr, 12, 144);

    CHECK(cpu_dst.IsIdentical(npp_res));
    CHECK(cpu_dstSqr.IsIdentical(npp_resSqr));
}

TEST_CASE("8uC1", "[NPP.Statistics.SqrIntegral32f]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size + 1, size + 1);
    cpu::Image<Pixel64fC1> cpu_dstSqr(size + 1, size + 1);
    cpu::Image<Pixel32fC1> npp_res(size + 1, size + 1);
    cpu::Image<Pixel64fC1> npp_resSqr(size + 1, size + 1);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size + 1, size + 1);
    nv::Image64fC1 npp_dstSqr(size + 1, size + 1);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.SqrIntegral(npp_dst, npp_dstSqr, 12, 144, nppCtx);
    npp_res << npp_dst;
    npp_resSqr << npp_dstSqr;

    cpu_src1.SqrIntegral(cpu_dst, cpu_dstSqr, 12, 144);

    CHECK(cpu_dst.IsIdentical(npp_res));
    CHECK(cpu_dstSqr.IsIdentical(npp_resSqr));
}
