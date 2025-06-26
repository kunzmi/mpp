#include <backends/cuda/devVar.h>
#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sc.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC2View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
#include <backends/npp/image/image16scC1View.h>
#include <backends/npp/image/image16scC2View.h>
#include <backends/npp/image/image16scC3View.h>
#include <backends/npp/image/image16scC4View.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC2View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image32fcC1View.h>
#include <backends/npp/image/image32fcC2View.h>
#include <backends/npp/image/image32fcC3View.h>
#include <backends/npp/image/image32fcC4View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sc.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32scC1View.h>
#include <backends/npp/image/image32scC2View.h>
#include <backends/npp/image/image32scC3View.h>
#include <backends/npp/image/image32scC4View.h>
#include <backends/npp/image/image32u.h>
#include <backends/npp/image/image32uC1View.h>
#include <backends/npp/image/image32uC2View.h>
#include <backends/npp/image/image32uC3View.h>
#include <backends/npp/image/image32uC4View.h>
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image64fC1View.h>
#include <backends/npp/image/image64fC2View.h>
#include <backends/npp/image/image64fC3View.h>
#include <backends/npp/image/image64fC4View.h>
#include <backends/npp/image/image8s.h>
#include <backends/npp/image/image8sC1View.h>
#include <backends/npp/image/image8sC2View.h>
#include <backends/npp/image/image8sC3View.h>
#include <backends/npp/image/image8sC4View.h>
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

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8uC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image8uC2 npp_src1(size, size);
    nv::Image8uC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8uC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8uC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8sC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel8sC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image8sC1 npp_src1(size, size);
    nv::Image8sC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8sC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel8sC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image8sC2 npp_src1(size, size);
    nv::Image8sC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8sC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel8sC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image8sC3 npp_src1(size, size);
    nv::Image8sC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8sC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image8sC4 npp_src1(size, size);
    nv::Image8sC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16uC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16uC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16uC2 npp_src1(size, size);
    nv::Image16uC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16uC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16uC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16sC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16sC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel16sC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16sC2 npp_src1(size, size);
    nv::Image16sC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16sC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16sC3 npp_src1(size, size);
    nv::Image16sC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16sC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16scC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image16scC1 npp_src1(size, size);
    nv::Image16scC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16scC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16scC2 npp_src1(size, size);
    nv::Image16scC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16scC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16scC3 npp_src1(size, size);
    nv::Image16scC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16scC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image16scC4 npp_src1(size, size);
    nv::Image16scC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32uC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32uC1::GetStreamContext();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel32uC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image32uC1 npp_src1(size, size);
    nv::Image32uC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32uC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32uC1::GetStreamContext();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel32uC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32uC2 npp_src1(size, size);
    nv::Image32uC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32uC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32uC1::GetStreamContext();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel32uC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32uC3 npp_src1(size, size);
    nv::Image32uC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32uC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32uC1::GetStreamContext();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel32uC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32uC4 npp_src1(size, size);
    nv::Image32uC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32sC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image32sC1 npp_src1(size, size);
    nv::Image32sC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32sC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32sC2 npp_src1(size, size);
    nv::Image32sC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32sC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32sC3 npp_src1(size, size);
    nv::Image32sC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32sC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32sC4 npp_src1(size, size);
    nv::Image32sC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32scC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32scC1::GetStreamContext();

    cpu::Image<Pixel32scC1> cpu_src1(size, size);
    cpu::Image<Pixel32scC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image32scC1 npp_src1(size, size);
    nv::Image32scC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32scC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32scC1::GetStreamContext();

    cpu::Image<Pixel32scC2> cpu_src1(size, size);
    cpu::Image<Pixel32scC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32scC2 npp_src1(size, size);
    nv::Image32scC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32scC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32scC1::GetStreamContext();

    cpu::Image<Pixel32scC3> cpu_src1(size, size);
    cpu::Image<Pixel32scC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32scC3 npp_src1(size, size);
    nv::Image32scC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32scC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32scC1::GetStreamContext();

    cpu::Image<Pixel32scC4> cpu_src1(size, size);
    cpu::Image<Pixel32scC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32scC4 npp_src1(size, size);
    nv::Image32scC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32fC2 npp_src1(size, size);
    nv::Image32fC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fcC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fcC1::GetStreamContext();

    cpu::Image<Pixel32fcC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image32fcC1 npp_src1(size, size);
    nv::Image32fcC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fcC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fcC1::GetStreamContext();

    cpu::Image<Pixel32fcC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32fcC2 npp_src1(size, size);
    nv::Image32fcC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fcC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fcC1::GetStreamContext();

    cpu::Image<Pixel32fcC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32fcC3 npp_src1(size, size);
    nv::Image32fcC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fcC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fcC1::GetStreamContext();

    cpu::Image<Pixel32fcC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image32fcC4 npp_src1(size, size);
    nv::Image32fcC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("64fC1", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image64fC1::GetStreamContext();

    cpu::Image<Pixel64fC1> cpu_src1(size, size);
    cpu::Image<Pixel64fC1> cpu_src2(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image64fC1 npp_src1(size, size);
    nv::Image64fC1 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("64fC2", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image64fC1::GetStreamContext();

    cpu::Image<Pixel64fC2> cpu_src1(size, size);
    cpu::Image<Pixel64fC2> cpu_src2(size, size);
    Pixel64fC2 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image64fC2 npp_src1(size, size);
    nv::Image64fC2 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("64fC3", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image64fC1::GetStreamContext();

    cpu::Image<Pixel64fC3> cpu_src1(size, size);
    cpu::Image<Pixel64fC3> cpu_src2(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image64fC3 npp_src1(size, size);
    nv::Image64fC3 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}

TEST_CASE("64fC4", "[NPP.Statistics.AverageError]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image64fC1::GetStreamContext();

    cpu::Image<Pixel64fC4> cpu_src1(size, size);
    cpu::Image<Pixel64fC4> cpu_src2(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res;
    nv::Image64fC4 npp_src1(size, size);
    nv::Image64fC4 npp_src2(size, size);
    mpp::cuda::DevVar<double> npp_dst(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.AverageErrorGetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    npp_src1.AverageError(npp_src2, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.AverageError(cpu_src2, cpu_dst, cpu_dstScalar);

    CHECK(cpu_dstScalar == Approx(npp_res).margin(0.00001));
}