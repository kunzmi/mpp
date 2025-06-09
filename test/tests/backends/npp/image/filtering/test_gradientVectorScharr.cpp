#include <backends/cuda/devVar.h>
#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <common/defines.h>
#include <common/safeCast.h>
#include <numbers>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Filtering.GradientVectorScharr]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    const uint seed            = Catch::getSeed();
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dstX(size, size);
    cpu::Image<Pixel16sC1> cpu_dstY(size, size);
    cpu::Image<Pixel16sC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel16sC1> npp_resX(size, size);
    cpu::Image<Pixel16sC1> npp_resY(size, size);
    cpu::Image<Pixel16sC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image16sC1 npp_dstX(size, size);
    nv::Image16sC1 npp_dstY(size, size);
    nv::Image16sC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 1));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 1));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1)); // it seems NPP doesn't do any rounding
}

TEST_CASE("8uC3", "[NPP.Filtering.GradientVectorScharr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dstX(size, size);
    cpu::Image<Pixel16sC1> cpu_dstY(size, size);
    cpu::Image<Pixel16sC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel16sC1> npp_resX(size, size);
    cpu::Image<Pixel16sC1> npp_resY(size, size);
    cpu::Image<Pixel16sC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image16sC1 npp_dstX(size, size);
    nv::Image16sC1 npp_dstY(size, size);
    nv::Image16sC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 1));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 1));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1)); // it seems NPP doesn't do any rounding
}

TEST_CASE("16sC1", "[NPP.Filtering.GradientVectorScharr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dstX(size, size);
    cpu::Image<Pixel32fC1> cpu_dstY(size, size);
    cpu::Image<Pixel32fC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel32fC1> npp_resX(size, size);
    cpu::Image<Pixel32fC1> npp_resY(size, size);
    cpu::Image<Pixel32fC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image32fC1 npp_dstX(size, size);
    nv::Image32fC1 npp_dstY(size, size);
    nv::Image32fC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 1));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 1));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.5f));
}

TEST_CASE("16sC3", "[NPP.Filtering.GradientVectorScharr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dstX(size, size);
    cpu::Image<Pixel32fC1> cpu_dstY(size, size);
    cpu::Image<Pixel32fC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel32fC1> npp_resX(size, size);
    cpu::Image<Pixel32fC1> npp_resY(size, size);
    cpu::Image<Pixel32fC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image16sC3 npp_src1(size, size);
    nv::Image32fC1 npp_dstX(size, size);
    nv::Image32fC1 npp_dstY(size, size);
    nv::Image32fC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 1));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 1));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.5f));
}

TEST_CASE("16uC1", "[NPP.Filtering.GradientVectorScharr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dstX(size, size);
    cpu::Image<Pixel32fC1> cpu_dstY(size, size);
    cpu::Image<Pixel32fC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel32fC1> npp_resX(size, size);
    cpu::Image<Pixel32fC1> npp_resY(size, size);
    cpu::Image<Pixel32fC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image32fC1 npp_dstX(size, size);
    nv::Image32fC1 npp_dstY(size, size);
    nv::Image32fC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 1));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 1));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.5f));
}

TEST_CASE("16uC3", "[NPP.Filtering.GradientVectorScharr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dstX(size, size);
    cpu::Image<Pixel32fC1> cpu_dstY(size, size);
    cpu::Image<Pixel32fC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel32fC1> npp_resX(size, size);
    cpu::Image<Pixel32fC1> npp_resY(size, size);
    cpu::Image<Pixel32fC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image32fC1 npp_dstX(size, size);
    nv::Image32fC1 npp_dstY(size, size);
    nv::Image32fC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 1));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 1));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 1));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.5f));
}

TEST_CASE("32fC1", "[NPP.Filtering.GradientVectorScharr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dstX(size, size);
    cpu::Image<Pixel32fC1> cpu_dstY(size, size);
    cpu::Image<Pixel32fC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel32fC1> npp_resX(size, size);
    cpu::Image<Pixel32fC1> npp_resY(size, size);
    cpu::Image<Pixel32fC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dstX(size, size);
    nv::Image32fC1 npp_dstY(size, size);
    nv::Image32fC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 0.00001f));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 0.00001f));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.00001f));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.0001f));
}

TEST_CASE("32fC3", "[NPP.Filtering.GradientVectorScharr]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dstX(size, size);
    cpu::Image<Pixel32fC1> cpu_dstY(size, size);
    cpu::Image<Pixel32fC1> cpu_dstMag(size, size);
    cpu::Image<Pixel32fC1> cpu_dstAngle(size, size);
    cpu::Image<Pixel32fC4> cpu_dstCoVar(size, size);
    cpu::Image<Pixel32fC1> npp_resX(size, size);
    cpu::Image<Pixel32fC1> npp_resY(size, size);
    cpu::Image<Pixel32fC1> npp_resMag(size, size);
    cpu::Image<Pixel32fC1> npp_resAngle(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dstX(size, size);
    nv::Image32fC1 npp_dstY(size, size);
    nv::Image32fC1 npp_dstMag(size, size);
    nv::Image32fC1 npp_dstAngle(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL1,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L1,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstX.IsSimilar(npp_resX, 0.00001f));
    CHECK(cpu_dstY.IsSimilar(npp_resY, 0.00001f));
    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.00001f));
    CHECK(cpu_dstAngle.IsSimilar(npp_resAngle, 0.0001f));

    npp_src1.GradientVectorScharrBorder(npp_dstX, npp_dstY, npp_dstMag, npp_dstAngle, NPP_MASK_SIZE_3_X_3, nppiNormL2,
                                        NPP_BORDER_REPLICATE, nppCtx);
    npp_resX << npp_dstX;
    npp_resY << npp_dstY;
    npp_resMag << npp_dstMag;
    npp_resAngle << npp_dstAngle;

    cpu_src1.GradientVectorScharr(cpu_dstX, cpu_dstY, cpu_dstMag, cpu_dstAngle, cpu_dstCoVar, Norm::L2,
                                  MaskSize::Mask_3x3, BorderType::Replicate);

    CHECK(cpu_dstMag.IsSimilar(npp_resMag, 0.0001f));
}