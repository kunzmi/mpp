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

TEST_CASE("8uC1", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 cpu_dstMax;
    byte npp_resMin;
    byte npp_resMax;
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(1);
    opp::cuda::DevVar<byte> npp_dstMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax);

    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
}

TEST_CASE("8uC3", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel8uC3 cpu_dstMin;
    Pixel8uC3 cpu_dstMax;
    byte cpu_dstScalarMin;
    byte cpu_dstScalarMax;
    byte npp_resMin[3];
    byte npp_resMax[3];
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(3);
    opp::cuda::DevVar<byte> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("8uC4", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    Pixel8uC4 cpu_dstMin;
    Pixel8uC4 cpu_dstMax;
    byte cpu_dstScalarMin;
    byte cpu_dstScalarMax;
    byte npp_resMin[4];
    byte npp_resMax[4];
    nv::Image8uC4 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(4);
    opp::cuda::DevVar<byte> npp_dstMax(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMin.w == npp_resMin[3]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
    CHECK(cpu_dstMax.w == npp_resMax[3]);
}

TEST_CASE("8uC4A", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel8uC4> cpu_src1(cpu_src1A);
    Pixel8uC4A cpu_dstMin;
    Pixel8uC4A cpu_dstMax;
    byte cpu_dstScalarMin;
    byte cpu_dstScalarMax;
    byte npp_resMin[3];
    byte npp_resMax[3];
    nv::Image8uC4 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(3);
    opp::cuda::DevVar<byte> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSizeA(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 10);

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxA(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1A.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("16uC1", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 cpu_dstMax;
    ushort npp_resMin;
    ushort npp_resMax;
    nv::Image16uC1 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(1);
    opp::cuda::DevVar<ushort> npp_dstMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 16000, 100);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax);

    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
}

TEST_CASE("16uC3", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel16uC3 cpu_dstMin;
    Pixel16uC3 cpu_dstMax;
    ushort cpu_dstScalarMin;
    ushort cpu_dstScalarMax;
    ushort npp_resMin[3];
    ushort npp_resMax[3];
    nv::Image16uC3 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(3);
    opp::cuda::DevVar<ushort> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 16000, 100);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("16uC4", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    Pixel16uC4 cpu_dstMin;
    Pixel16uC4 cpu_dstMax;
    ushort cpu_dstScalarMin;
    ushort cpu_dstScalarMax;
    ushort npp_resMin[4];
    ushort npp_resMax[4];
    nv::Image16uC4 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(4);
    opp::cuda::DevVar<ushort> npp_dstMax(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 16000, 100);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMin.w == npp_resMin[3]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
    CHECK(cpu_dstMax.w == npp_resMax[3]);
}

TEST_CASE("16uC4A", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel16uC4> cpu_src1(cpu_src1A);
    Pixel16uC4A cpu_dstMin;
    Pixel16uC4A cpu_dstMax;
    ushort cpu_dstScalarMin;
    ushort cpu_dstScalarMax;
    ushort npp_resMin[3];
    ushort npp_resMax[3];
    nv::Image16uC4 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(3);
    opp::cuda::DevVar<ushort> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSizeA(nppCtx));

    cpu_src1.FillRandomNormal(seed, 16000, 100);

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxA(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1A.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("16sC1", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    Pixel16sC1 cpu_dstMin;
    Pixel16sC1 cpu_dstMax;
    short npp_resMin;
    short npp_resMax;
    nv::Image16sC1 npp_src1(size, size);
    opp::cuda::DevVar<short> npp_dstMin(1);
    opp::cuda::DevVar<short> npp_dstMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax);

    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
}

TEST_CASE("16sC3", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    Pixel16sC3 cpu_dstMin;
    Pixel16sC3 cpu_dstMax;
    short cpu_dstScalarMin;
    short cpu_dstScalarMax;
    short npp_resMin[3];
    short npp_resMax[3];
    nv::Image16sC3 npp_src1(size, size);
    opp::cuda::DevVar<short> npp_dstMin(3);
    opp::cuda::DevVar<short> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("16sC4", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    Pixel16sC4 cpu_dstMin;
    Pixel16sC4 cpu_dstMax;
    short cpu_dstScalarMin;
    short cpu_dstScalarMax;
    short npp_resMin[4];
    short npp_resMax[4];
    nv::Image16sC4 npp_src1(size, size);
    opp::cuda::DevVar<short> npp_dstMin(4);
    opp::cuda::DevVar<short> npp_dstMax(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMin.w == npp_resMin[3]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
    CHECK(cpu_dstMax.w == npp_resMax[3]);
}

TEST_CASE("16sC4A", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel16sC4> cpu_src1(cpu_src1A);
    Pixel16sC4A cpu_dstMin;
    Pixel16sC4A cpu_dstMax;
    short cpu_dstScalarMin;
    short cpu_dstScalarMax;
    short npp_resMin[3];
    short npp_resMax[3];
    nv::Image16sC4 npp_src1(size, size);
    opp::cuda::DevVar<short> npp_dstMin(3);
    opp::cuda::DevVar<short> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSizeA(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxA(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1A.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("32fC1", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 cpu_dstMax;
    float npp_resMin;
    float npp_resMax;
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dstMin(1);
    opp::cuda::DevVar<float> npp_dstMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax);

    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
}

TEST_CASE("32fC3", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel32fC3 cpu_dstMin;
    Pixel32fC3 cpu_dstMax;
    float cpu_dstScalarMin;
    float cpu_dstScalarMax;
    float npp_resMin[3];
    float npp_resMax[3];
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dstMin(3);
    opp::cuda::DevVar<float> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("32fC4", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    Pixel32fC4 cpu_dstMin;
    Pixel32fC4 cpu_dstMax;
    float cpu_dstScalarMin;
    float cpu_dstScalarMax;
    float npp_resMin[4];
    float npp_resMax[4];
    nv::Image32fC4 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dstMin(4);
    opp::cuda::DevVar<float> npp_dstMax(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMax(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMin.w == npp_resMin[3]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
    CHECK(cpu_dstMax.w == npp_resMax[3]);
}

TEST_CASE("32fC4A", "[NPP.Statistics.MinMax]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel32fC4> cpu_src1(cpu_src1A);
    Pixel32fC4A cpu_dstMin;
    Pixel32fC4A cpu_dstMax;
    float cpu_dstScalarMin;
    float cpu_dstScalarMax;
    float npp_resMin[3];
    float npp_resMax[3];
    nv::Image32fC4 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dstMin(3);
    opp::cuda::DevVar<float> npp_dstMax(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxGetBufferHostSizeA(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 1000);

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxA(npp_dstMin, npp_dstMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;

    cpu_src1A.MinMax(cpu_dstMin, cpu_dstMax, cpu_dstScalarMin, cpu_dstScalarMax);

    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}
