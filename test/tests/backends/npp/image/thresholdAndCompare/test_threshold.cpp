#include <backends/cuda/devVar.h>
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
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
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

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, 127, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(127, CompareOp::Less, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, 127, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(127, CompareOp::Greater, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(127, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold(127, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, {127, 120, 130}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Less, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, {127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Greater, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold({127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::ImageView<Pixel8uC4A> cpu_dstA(cpu_dst);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    npp_src1.ThresholdA(npp_dst, {127, 120, 130}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Less, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA(npp_dst, {127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Greater, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA({127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16sC1", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, 127, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(127, CompareOp::Less, cpu_dst);

    // BUG in NPP! signed short is treates as unsigned short, so all negative values are large positive values...
    // CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, 127, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(127, CompareOp::Greater, cpu_dst);

    // CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(127, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold(127, CompareOp::Greater);

    // CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16sC3", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> npp_res(size, size);
    nv::Image16sC3 npp_src1(size, size);
    nv::Image16sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, {127, 120, 130}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Less, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, {127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Greater, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold({127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16sC4A", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    cpu::ImageView<Pixel16sC4A> cpu_src1A(cpu_src1);
    cpu::ImageView<Pixel16sC4A> cpu_dstA(cpu_dst);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    npp_src1.ThresholdA(npp_dst, {127, 120, 130}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Less, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA(npp_dst, {127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Greater, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA({127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, 127, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(127, CompareOp::Less, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, 127, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(127, CompareOp::Greater, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(127, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold(127, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, {127, 120, 130}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Less, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, {127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Greater, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold({127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold({127, 120, 130}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A(cpu_src1);
    cpu::ImageView<Pixel16uC4A> cpu_dstA(cpu_dst);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    npp_src1.ThresholdA(npp_dst, {127, 120, 130}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Less, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA(npp_dst, {127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Greater, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA({127, 120, 130}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1A.Threshold({127, 120, 130}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, 0.5f, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(0.5f, CompareOp::Less, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, 0.5f, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold(0.5f, CompareOp::Greater, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(0.5f, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold(0.5f, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Threshold(npp_dst, {0.5f, 0.55f, 0.45f}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({0.5f, 0.55f, 0.45f}, CompareOp::Less, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold(npp_dst, {0.5f, 0.55f, 0.45f}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Threshold({0.5f, 0.55f, 0.45f}, CompareOp::Greater, cpu_dst);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.Threshold({0.5f, 0.55f, 0.45f}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Threshold({0.5f, 0.55f, 0.45f}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Statistics.Threshold]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A(cpu_src1);
    cpu::ImageView<Pixel32fC4A> cpu_dstA(cpu_dst);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_dst.Set(255);

    cpu_src1 >> npp_src1;
    cpu_dst >> npp_dst;

    npp_src1.ThresholdA(npp_dst, {0.5f, 0.55f, 0.45f}, NPP_CMP_LESS, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({0.5f, 0.55f, 0.45f}, CompareOp::Less, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA(npp_dst, {0.5f, 0.55f, 0.45f}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Threshold({0.5f, 0.55f, 0.45f}, CompareOp::Greater, cpu_dstA);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.ThresholdA({0.5f, 0.55f, 0.45f}, NPP_CMP_GREATER, nppCtx);
    npp_res << npp_src1;

    cpu_src1A.Threshold({0.5f, 0.55f, 0.45f}, CompareOp::Greater);

    CHECK(cpu_src1.IsIdentical(npp_res));
}
