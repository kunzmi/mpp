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
#include <common/image/affineTransformation.h>
#include <common/image/bound.h>
#include <common/image/quad.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC3", "[NPP.GeometricTransforms.Resize]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1 >> npp_src1;

    Vec2d scaleFactor(2, 3);
    Vec2d shift(-128, -128);

    // NearestNeighbor interpolation
    npp_src1.ResizeSqrPixel(npp_dst, scaleFactor.x, scaleFactor.y, shift.x, shift.y, static_cast<int>(NPPI_INTER_NN),
                            nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shift, InterpolationMode::NearestNeighbor, BorderType::None);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // linear interpolation
    npp_src1.ResizeSqrPixel(npp_dst, scaleFactor.x, scaleFactor.y, shift.x, shift.y,
                            static_cast<int>(NPPI_INTER_LINEAR), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shift, InterpolationMode::Linear, BorderType::None);

    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(maxErr == 1);
    CHECK(summedErr < 4000); // different rounding mode for linear interpolation in NPP?

    // cubic interpolation
    npp_src1.ResizeSqrPixel(npp_dst, scaleFactor.x, scaleFactor.y, shift.x, shift.y, static_cast<int>(NPPI_INTER_CUBIC),
                            nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shift, InterpolationMode::CubicLagrange, BorderType::None);

    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(maxErr == 1);
    CHECK(summedErr < 72);

    scaleFactor = Vec2d(0.5, 0.3);
    shift       = Vec2d(64, 64);

    // NearestNeighbor interpolation
    npp_dst.Set({128}, nppCtx);
    cpu_dst.Set({128});
    npp_src1.ResizeSqrPixel(npp_dst, scaleFactor.x, scaleFactor.y, shift.x, shift.y, static_cast<int>(NPPI_INTER_NN),
                            nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shift, InterpolationMode::NearestNeighbor, BorderType::None);

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);
    CHECK(summedErr == 0);

    // linear interpolation
    npp_dst.Set({128}, nppCtx);
    cpu_dst.Set({128});
    npp_src1.ResizeSqrPixel(npp_dst, scaleFactor.x, scaleFactor.y, shift.x, shift.y,
                            static_cast<int>(NPPI_INTER_LINEAR), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shift, InterpolationMode::Linear, BorderType::None);

    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(maxErr == 1);
    CHECK(summedErr < 800); // different rounding mode for linear interpolation in NPP?

    // cubic interpolation
    npp_dst.Set({128}, nppCtx);
    cpu_dst.Set({128});
    npp_src1.ResizeSqrPixel(npp_dst, scaleFactor.x, scaleFactor.y, shift.x, shift.y, static_cast<int>(NPPI_INTER_CUBIC),
                            nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shift, InterpolationMode::CubicLagrange, BorderType::None);

    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(maxErr == 1);
    CHECK(summedErr < 9);
}

TEST_CASE("8uC3 - downscale", "[NPP.GeometricTransforms.Resize]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size / 3, size / 2);
    cpu::Image<Pixel8uC3> npp_res(size / 3, size / 2);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size / 3, size / 2);

    cpu_src1 >> npp_src1;

    const Vec2d scaleFactor = Vec2d(cpu_dst.SizeRoi()) / Vec2d(cpu_src1.SizeRoi());
    const Vec2d shiftForNPP = cpu_src1.ResizeGetNPPShift(cpu_dst);

    // NearestNeighbor interpolation
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shiftForNPP, InterpolationMode::NearestNeighbor, BorderType::None);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // super sampling
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_SUPER), nppCtx);
    npp_res << npp_dst;

    // no shift at all when using super sampling, and only for 8u???
    cpu_src1.Resize(cpu_dst, scaleFactor, {0}, InterpolationMode::Super, BorderType::None);

    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    /*cpu_dst.Save(root / "scaleMPP.tif");
    npp_res.Save(root / "scaleNPP.tif");*/

    CHECK(maxErr == 1);
    CHECK(summedErr < 48);
}
TEST_CASE("32fC3 - downscale", "[NPP.GeometricTransforms.Resize]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC3> cpu_src = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size / 3, size / 2);
    cpu::Image<Pixel32fC3> npp_res(size / 3, size / 2);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size / 3, size / 2);

    cpu_src1 >> npp_src1;

    const Vec2d scaleFactor = Vec2d(cpu_dst.SizeRoi()) / Vec2d(cpu_src1.SizeRoi());
    const Vec2d shiftForNPP = cpu_src1.ResizeGetNPPShift(cpu_dst);

    // NearestNeighbor interpolation
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shiftForNPP, InterpolationMode::NearestNeighbor, BorderType::None);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // super sampling
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_SUPER), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shiftForNPP, InterpolationMode::Super, BorderType::None);

    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(maxErr == 0);
    CHECK(summedErr == 0);
}

TEST_CASE("8uC3 - upscale", "[NPP.GeometricTransforms.Resize]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size * 3, size * 2);
    cpu::Image<Pixel8uC3> npp_res(size * 3, size * 2);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size * 3, size * 2);

    cpu_src1 >> npp_src1;

    const Vec2d scaleFactor = Vec2d(cpu_dst.SizeRoi()) / Vec2d(cpu_src1.SizeRoi());
    const Vec2d shiftForNPP = cpu_src1.ResizeGetNPPShift(cpu_dst);

    // NearestNeighbor interpolation
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shiftForNPP, InterpolationMode::NearestNeighbor, BorderType::None);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // super sampling
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_CUBIC), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shiftForNPP, InterpolationMode::CubicLagrange, BorderType::None);

    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(maxErr == 1);
    CHECK(summedErr < 22);
}
TEST_CASE("32fC3 - upscale", "[NPP.GeometricTransforms.Resize]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC3> cpu_src = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size * 3, size * 2);
    cpu::Image<Pixel32fC3> npp_res(size * 3, size * 2);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size * 3, size * 2);

    cpu_src1 >> npp_src1;

    const Vec2d scaleFactor = Vec2d(cpu_dst.SizeRoi()) / Vec2d(cpu_src1.SizeRoi());
    const Vec2d shiftForNPP = cpu_src1.ResizeGetNPPShift(cpu_dst);

    // NearestNeighbor interpolation
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shiftForNPP, InterpolationMode::NearestNeighbor, BorderType::None);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // super sampling
    npp_src1.Resize(npp_dst, static_cast<int>(NPPI_INTER_CUBIC), nppCtx);
    npp_res << npp_dst;

    cpu_src1.Resize(cpu_dst, scaleFactor, shiftForNPP, InterpolationMode::CubicLagrange, BorderType::None);

    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffInf(npp_res, maxErr, maxErrChannel);
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(maxErr == 0);
    CHECK(summedErr == 0);
}

TEST_CASE("8uP3", "[NPP.GeometricTransforms.Resize]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1 >> npp_src1;

    cpu::Image<Pixel8uC1> cpu_srcC1(size, size);
    cpu::Image<Pixel8uC1> cpu_srcC2(size, size);
    cpu::Image<Pixel8uC1> cpu_srcC3(size, size);

    cpu_src1.Copy(cpu_srcC1, cpu_srcC2, cpu_srcC3);

    cpu::Image<Pixel8uC1> cpu_dstC1(size, size);
    cpu::Image<Pixel8uC1> cpu_dstC2(size, size);
    cpu::Image<Pixel8uC1> cpu_dstC3(size, size);

    Vec2d scaleFactor(2, 3);
    Vec2d shift(-128, -128);

    // NearestNeighbor interpolation
    npp_src1.ResizeSqrPixel(npp_dst, scaleFactor.x, scaleFactor.y, shift.x, shift.y, static_cast<int>(NPPI_INTER_NN),
                            nppCtx);
    npp_res << npp_dst;

    cpu::Image<Pixel8uC3>::Resize(cpu_srcC1, cpu_srcC2, cpu_srcC3, cpu_dstC1, cpu_dstC2, cpu_dstC3, scaleFactor, shift,
                                  InterpolationMode::NearestNeighbor, BorderType::None);
    cpu::Image<Pixel8uC3>::Copy(cpu_dstC1, cpu_dstC2, cpu_dstC3, cpu_dst);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);
}