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
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC2View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

// If result type is byte (unsigned char), then NPP handles seperatly the case that src1 == 0 and src2 == 0. Any
// number devided by 0 in floating point results in INF, but 0 / 0 gets NAN. Usually this would flush to 0 while
// casting to integer, but NPP returns 255...
// Turns out that IPP behaves normally, so MPP also avoids all these "if pixel is 0"s just for 8u data type and we fix
// the result images for comparison:
void fixZeroDivision(cpu::Image<Pixel8uC1> &aSrc1, cpu::Image<Pixel8uC1> &aSrc2, cpu::Image<Pixel8uC1> &aDst)
{
    auto iter2   = aSrc2.begin();
    auto iterDst = aDst.begin();

    for (auto &iter1 : aSrc1)
    {
        if (iter1.Value() == 0 && iter2.Value() == 0 && iterDst.Value() == 255)
        {
            iterDst.Value() = 0;
        }
        iter2++;
        iterDst++;
    }
}

void fixZeroDivision(cpu::Image<Pixel8uC3> &aSrc1, cpu::Image<Pixel8uC3> &aSrc2, cpu::Image<Pixel8uC3> &aDst)
{
    auto iter2   = aSrc2.begin();
    auto iterDst = aDst.begin();

    for (auto &iter1 : aSrc1)
    {
        if (iter1.Value().x == 0 && iter2.Value().x == 0 && iterDst.Value().x == 255)
        {
            iterDst.Value().x = 0;
        }
        if (iter1.Value().y == 0 && iter2.Value().y == 0 && iterDst.Value().y == 255)
        {
            iterDst.Value().y = 0;
        }
        if (iter1.Value().z == 0 && iter2.Value().z == 0 && iterDst.Value().z == 255)
        {
            iterDst.Value().z = 0;
        }
        iter2++;
        iterDst++;
    }
}

void fixZeroDivision(cpu::Image<Pixel8uC4> &aSrc1, cpu::Image<Pixel8uC4> &aSrc2, cpu::Image<Pixel8uC4> &aDst)
{
    auto iter2   = aSrc2.begin();
    auto iterDst = aDst.begin();

    for (auto &iter1 : aSrc1)
    {
        if (iter1.Value().x == 0 && iter2.Value().x == 0 && iterDst.Value().x == 255)
        {
            iterDst.Value().x = 0;
        }
        if (iter1.Value().y == 0 && iter2.Value().y == 0 && iterDst.Value().y == 255)
        {
            iterDst.Value().y = 0;
        }
        if (iter1.Value().z == 0 && iter2.Value().z == 0 && iterDst.Value().z == 255)
        {
            iterDst.Value().z = 0;
        }
        if (iter1.Value().w == 0 && iter2.Value().w == 0 && iterDst.Value().w == 255)
        {
            iterDst.Value().w = 0;
        }
        iter2++;
        iterDst++;
    }
}

TEST_CASE("8uC1", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_src2(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_src2(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;
    fixZeroDivision(cpu_src1, cpu_src2, npp_res);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    nv::Image8uC4 npp_dst(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel8uC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel8uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::TowardZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, -2, RoundingMode::TowardZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_RoundA(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_RoundA(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_src2(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_src2(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    nv::Image16uC4 npp_dst(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16uC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel16uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::TowardZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, -2, RoundingMode::TowardZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_RoundA(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_RoundA(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC1", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_src2(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC1 npp_src2(size, size);
    nv::Image16sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC3", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_src2(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> npp_res(size, size);
    nv::Image16sC3 npp_src1(size, size);
    nv::Image16sC3 npp_src2(size, size);
    nv::Image16sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC4", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_src2(size, size);
    nv::Image16sC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2, RoundingMode::TowardZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Div(cpu_src2, cpu_dst, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_Round(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Div(cpu_src2, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_Round(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16sC4A", "[NPP.Arithmetic.DivRound]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_src2(size, size);
    nv::Image16sC4 npp_dst(size, size);
    cpu::ImageView<Pixel16sC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16sC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel16sC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::TowardZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, -2, RoundingMode::TowardZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_TOWARD_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::NearestTiesToEven);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, -2, RoundingMode::NearestTiesToEven);
    npp_dst.Div_RoundA(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_TO_EVEN, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Div(cpu_src2A, cpu_dstA, 0, RoundingMode::NearestTiesAwayFromZero);
    npp_src2.Div_RoundA(npp_src1, npp_dst, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Div(cpu_src2A, -2, RoundingMode::NearestTiesAwayFromZero);
    npp_dst.Div_RoundA(npp_src2, NppRoundMode::NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO, -2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}