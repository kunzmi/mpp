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

TEST_CASE("8uC1", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel8uC1 cpu_dst;
    byte npp_res;
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst);

    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("8uC3", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel8uC3 cpu_dst;
    byte cpu_dstScalar;
    byte npp_res[3];
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("8uC4", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    Pixel8uC4 cpu_dst;
    byte cpu_dstScalar;
    byte npp_res[4];
    nv::Image8uC4 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("16uC1", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel16uC1 cpu_dst;
    ushort npp_res;
    nv::Image16uC1 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 16000, 100);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst);

    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("16uC3", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel16uC3 cpu_dst;
    ushort cpu_dstScalar;
    ushort npp_res[3];
    nv::Image16uC3 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 16000, 100);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("16uC4", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    Pixel16uC4 cpu_dst;
    ushort cpu_dstScalar;
    ushort npp_res[4];
    nv::Image16uC4 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 16000, 100);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("16sC1", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    Pixel16sC1 cpu_dst;
    short npp_res;
    nv::Image16sC1 npp_src1(size, size);
    opp::cuda::DevVar<short> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst);

    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("16sC3", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    Pixel16sC3 cpu_dst;
    short cpu_dstScalar;
    short npp_res[3];
    nv::Image16sC3 npp_src1(size, size);
    opp::cuda::DevVar<short> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("16sC4", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    Pixel16sC4 cpu_dst;
    short cpu_dstScalar;
    short npp_res[4];
    nv::Image16sC4 npp_src1(size, size);
    opp::cuda::DevVar<short> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("32fC1", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel32fC1 cpu_dst;
    float npp_res;
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst);

    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("32fC3", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel32fC3 cpu_dst;
    float cpu_dstScalar;
    float npp_res[3];
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("32fC4", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    Pixel32fC4 cpu_dst;
    float cpu_dstScalar;
    float npp_res[4];
    nv::Image32fC4 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.Min(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("32fC4A", "[NPP.Statistics.Min]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel32fC4> cpu_src1(cpu_src1A);
    Pixel32fC4A cpu_dst;
    float cpu_dstScalar;
    float npp_res[4];
    nv::Image32fC4 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinGetBufferHostSizeA(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinA(npp_buffer, npp_dst, nppCtx);
    npp_dst >> npp_res;

    cpu_src1A.Min(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}
