#include <backends/cuda/devVar.h>
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

TEST_CASE("8uC1", "[NPP.Statistics.PSNR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    float npp_res;
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_src2(size, size);
    opp::cuda::DevVar<float> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.PSNRGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.PSNR(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.PSNR(cpu_src2, cpu_dst, 255);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8uC3", "[NPP.Statistics.PSNR]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    float npp_res[3];
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_src2(size, size);
    opp::cuda::DevVar<float> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.PSNRGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.PSNR(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.PSNR(cpu_src2, cpu_dst, cpu_dstScalar, 255);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}
