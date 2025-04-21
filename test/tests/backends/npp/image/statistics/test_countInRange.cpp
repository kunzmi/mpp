#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
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
#include <common/safeCast.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.CountInRange]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel64uC1 cpu_dst;
    int npp_res;
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<int> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.CountInRangeGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.CountInRange(npp_dst, 130, 140, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.CountInRange(130, 140, cpu_dst);

    CHECK(cpu_dst.x == to_size_t(npp_res));
}

TEST_CASE("8uC3", "[NPP.Statistics.CountInRange]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel64uC3 cpu_dst;
    size_t cpu_dstScalar;
    int npp_res[3];
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<int> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.CountInRangeGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;
    Pixel8uC3 lowerLimit(130, 120, 110);
    Pixel8uC3 upperLimit(140, 150, 160);
    npp_src1.CountInRange(npp_dst, lowerLimit, upperLimit, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.CountInRange(lowerLimit, upperLimit, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == to_size_t(npp_res[0]));
    CHECK(cpu_dst.y == to_size_t(npp_res[1]));
    CHECK(cpu_dst.z == to_size_t(npp_res[2]));
}

TEST_CASE("8uC4", "[NPP.Statistics.CountInRange]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel8uC4> cpu_src1(cpu_src1A);
    Pixel64uC4A cpu_dst;
    size_t cpu_dstScalar;
    int npp_res[3];
    nv::Image8uC4 npp_src1(size, size);
    opp::cuda::DevVar<int> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.CountInRangeGetBufferHostSizeA(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    Pixel8uC3 lowerLimit(130, 120, 110);
    Pixel8uC3 upperLimit(140, 150, 160);
    npp_src1.CountInRangeA(npp_dst, lowerLimit, upperLimit, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1A.CountInRange(lowerLimit, upperLimit, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == to_size_t(npp_res[0]));
    CHECK(cpu_dst.y == to_size_t(npp_res[1]));
    CHECK(cpu_dst.z == to_size_t(npp_res[2]));
}

TEST_CASE("32fC1", "[NPP.Statistics.CountInRange]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel64uC1 cpu_dst;
    int npp_res;
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<int> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.CountInRangeGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.CountInRange(npp_dst, 130, 140, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.CountInRange(130, 140, cpu_dst);

    CHECK(cpu_dst.x == to_size_t(npp_res));
}

TEST_CASE("32fC3", "[NPP.Statistics.CountInRange]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel64uC3 cpu_dst;
    size_t cpu_dstScalar;
    int npp_res[3];
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<int> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.CountInRangeGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    Pixel32fC3 lowerLimit(130, 120, 110);
    Pixel32fC3 upperLimit(140, 150, 160);
    npp_src1.CountInRange(npp_dst, lowerLimit, upperLimit, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.CountInRange(lowerLimit, upperLimit, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == to_size_t(npp_res[0]));
    CHECK(cpu_dst.y == to_size_t(npp_res[1]));
    CHECK(cpu_dst.z == to_size_t(npp_res[2]));
}

TEST_CASE("32fC4A", "[NPP.Statistics.CountInRange]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel32fC4> cpu_src1(cpu_src1A);
    Pixel64uC4A cpu_dst;
    size_t cpu_dstScalar;
    int npp_res[3];
    nv::Image32fC4 npp_src1(size, size);
    opp::cuda::DevVar<int> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.CountInRangeGetBufferHostSizeA(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    Pixel32fC3 lowerLimit(130, 120, 110);
    Pixel32fC3 upperLimit(140, 150, 160);
    npp_src1.CountInRangeA(npp_dst, lowerLimit, upperLimit, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1A.CountInRange(lowerLimit, upperLimit, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == to_size_t(npp_res[0]));
    CHECK(cpu_dst.y == to_size_t(npp_res[1]));
    CHECK(cpu_dst.z == to_size_t(npp_res[2]));
}
