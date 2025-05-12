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

constexpr int size = 128;
constexpr Vector2<int> border(69, 57);
constexpr Vector2<float> subpixeDelta(0.23f, 0.34f);

TEST_CASE("8uC1", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel8uC1> npp_res(4 * size, 4 * size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel8uC3> npp_res(4 * size, 4 * size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel8uC4> npp_res(4 * size, 4 * size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel8uC4> cpu_dst(4 * size, 4 * size);
    cpu::ImageView<Pixel8uC4A> cpu_dstA(cpu_dst);
    cpu::Image<Pixel8uC4> npp_res(4 * size, 4 * size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);
    npp_dst.Set(128, nppCtx);
    cpu_dst.Set(128);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorderA(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorderA(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorderA(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel16uC1> npp_res(4 * size, 4 * size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel16uC3> npp_res(4 * size, 4 * size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel16uC4> npp_res(4 * size, 4 * size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel16uC4> cpu_dst(4 * size, 4 * size);
    cpu::ImageView<Pixel16uC4A> cpu_dstA(cpu_dst);
    cpu::Image<Pixel16uC4> npp_res(4 * size, 4 * size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);
    npp_dst.Set(128, nppCtx);
    cpu_dst.Set(128);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorderA(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorderA(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorderA(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel32fC1> npp_res(4 * size, 4 * size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel32fC3> npp_res(4 * size, 4 * size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(4 * size, 4 * size);
    cpu::Image<Pixel32fC4> npp_res(4 * size, 4 * size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorder(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorder(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.DataExchangeAndInit.CopyBorder]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A(cpu_src1);
    cpu::Image<Pixel32fC4> cpu_dst(4 * size, 4 * size);
    cpu::ImageView<Pixel32fC4A> cpu_dstA(cpu_dst);
    cpu::Image<Pixel32fC4> npp_res(4 * size, 4 * size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(4 * size, 4 * size);

    cpu_src1.FillRandom(seed);
    npp_dst.Set(128, nppCtx);
    cpu_dst.Set(128);

    cpu_src1 >> npp_src1;

    npp_src1.CopyWrapBorderA(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Wrap);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyConstBorderA(npp_dst, border.y, border.x, 16, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Constant, 16);

    CHECK(cpu_dst.IsIdentical(npp_res));

    npp_src1.CopyReplicateBorderA(npp_dst, border.y, border.x, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.Copy(cpu_dstA, border, BorderType::Replicate);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC1", "[NPP.DataExchangeAndInit.CopySubPix]")
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

    npp_src1.CopySubpix(npp_dst, subpixeDelta.x, subpixeDelta.y, nppCtx);
    npp_res << npp_dst;

    cpu_src1.Copy(cpu_dst, subpixeDelta, InterpolationMode::Linear);

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}