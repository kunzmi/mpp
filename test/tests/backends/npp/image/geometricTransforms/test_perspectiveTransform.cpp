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
#include <common/image/matrix.h>
#include <common/image/quad.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

TEST_CASE("8uC3", "[NPP.GeometricTransforms.Perspective]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC1> cpu_mask     = cpu::Image<Pixel8uC1>::Load(root / "knownErrorsNPPperspective.tif");
    cpu::Image<Pixel8uC1> cpu_maskBack = cpu::Image<Pixel8uC1>::Load(root / "knownErrorsNPPperspectiveBack.tif");
    cpu::Image<Pixel8uC3> cpu_src1     = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu_mask.SubInv(255);
    cpu_maskBack.SubInv(255);

    const int size = cpu_src1.Width();
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> cpu_dst2(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    cpu::Image<Pixel8uC3> npp_res2(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);
    nv::Image8uC3 npp_dst2(size, size);

    cpu_src1 >> npp_src1;

    AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation(Vec2d(-size / 2));
    AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
    AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation(Vec2d(size / 2));
    AffineTransformation<double> affine = shift2 * rot * shift1;
    PerspectiveTransformation<double> perspective(affine);
    perspective(2, 0) = 0.0002;
    perspective(2, 1) = -0.0003;

    // NearestNeighbor interpolation
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_dst2.Set({0, 128, 0}, nppCtx);
    npp_src1.WarpPerspective(npp_dst, perspective, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_dst.WarpPerspectiveBack(npp_dst2, perspective, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;
    npp_res2 << npp_dst2;

    cpu_dst.Set({128, 0, 0});
    cpu_dst2.Set({0, 128, 0});

    cpu_src1.WarpPerspective(cpu_dst, perspective, InterpolationMode::NearestNeighbor, BorderType::None);
    cpu_dst.WarpPerspectiveBack(cpu_dst2, perspective, InterpolationMode::NearestNeighbor, BorderType::None);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1Masked(npp_res, summedErr, summedErrChannel, cpu_mask);

    CHECK(summedErr == Vec3d{6, 3, 0}); // one pixel differs a bit

    cpu_dst2.NormDiffL1Masked(npp_res2, summedErr, summedErrChannel, cpu_maskBack);

    CHECK(summedErr == Vec3d{0, 0, 0});

    // linear interpolation
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.WarpPerspective(npp_dst, perspective, static_cast<int>(NPPI_INTER_LINEAR), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});

    cpu_src1.WarpPerspective(cpu_dst, perspective, InterpolationMode::Linear, BorderType::None);

    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffInfMasked(npp_res, maxErr, maxErrChannel, cpu_mask);
    cpu_dst.NormDiffL1Masked(npp_res, summedErr, summedErrChannel, cpu_mask);

    CHECK(maxErr == 1);
    CHECK(summedErr < 8); // less than 8 pixels differ by max 1

    // cubic interpolation
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.WarpPerspective(npp_dst, perspective, static_cast<int>(NPPI_INTER_CUBIC), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});

    cpu_src1.WarpPerspective(cpu_dst, perspective, InterpolationMode::CubicLagrange, BorderType::None);

    cpu_dst.NormDiffInfMasked(npp_res, maxErr, maxErrChannel, cpu_mask);
    cpu_dst.NormDiffL1Masked(npp_res, summedErr, summedErrChannel, cpu_mask);

    CHECK(maxErr == 1);
    CHECK(summedErr < 9); // less than 9 pixels differ by max 1
}

TEST_CASE("32fC3", "[NPP.GeometricTransforms.Perspective]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC1> cpu_mask     = cpu::Image<Pixel8uC1>::Load(root / "knownErrorsNPPperspective.tif");
    cpu::Image<Pixel8uC1> cpu_maskBack = cpu::Image<Pixel8uC1>::Load(root / "knownErrorsNPPperspectiveBack.tif");
    cpu::Image<Pixel8uC3> cpu_src      = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu_mask.SubInv(255);
    cpu_maskBack.SubInv(255);

    const int size = cpu_src.Width();
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> cpu_dst2(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    cpu::Image<Pixel32fC3> npp_res2(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);
    nv::Image32fC3 npp_dst2(size, size);

    cpu_src.Convert(cpu_src1);
    cpu_src1 >> npp_src1;

    AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation(Vec2d(-size / 2));
    AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
    AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation(Vec2d(size / 2));
    AffineTransformation<double> affine = shift2 * rot * shift1;
    PerspectiveTransformation<double> perspective(affine);
    perspective(2, 0) = 0.0002;
    perspective(2, 1) = -0.0003;

    // NearestNeighbor interpolation
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_dst2.Set({0, 128, 0}, nppCtx);
    npp_src1.WarpPerspective(npp_dst, perspective, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_dst.WarpPerspectiveBack(npp_dst2, perspective, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;
    npp_res2 << npp_dst2;

    cpu_dst.Set({128, 0, 0});
    cpu_dst2.Set({0, 128, 0});

    cpu_src1.WarpPerspective(cpu_dst, perspective, InterpolationMode::NearestNeighbor, BorderType::None);
    cpu_dst.WarpPerspectiveBack(cpu_dst2, perspective, InterpolationMode::NearestNeighbor, BorderType::None);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1Masked(npp_res, summedErr, summedErrChannel, cpu_mask);

    CHECK(summedErr == Vec3d{6, 3, 0}); // one pixel differs a bit

    cpu_dst2.NormDiffL1Masked(npp_res2, summedErr, summedErrChannel, cpu_maskBack);

    CHECK(summedErr == Vec3d{0, 0, 0});

    // linear interpolation
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.WarpPerspective(npp_dst, perspective, static_cast<int>(NPPI_INTER_LINEAR), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});

    cpu_src1.WarpPerspective(cpu_dst, perspective, InterpolationMode::Linear, BorderType::None);

    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffInfMasked(npp_res, maxErr, maxErrChannel, cpu_mask);
    cpu_dst.NormDiffL1Masked(npp_res, summedErr, summedErrChannel, cpu_mask);

    CHECK(maxErr < 0.0035);
    CHECK(summedErr < 6);

    // cubic interpolation
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.WarpPerspective(npp_dst, perspective, static_cast<int>(NPPI_INTER_CUBIC), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});

    cpu_src1.WarpPerspective(cpu_dst, perspective, InterpolationMode::CubicLagrange, BorderType::None);

    cpu_dst.NormDiffInfMasked(npp_res, maxErr, maxErrChannel, cpu_mask);
    cpu_dst.NormDiffL1Masked(npp_res, summedErr, summedErrChannel, cpu_mask);

    CHECK(maxErr < 0.004);
    CHECK(summedErr < 7);
}

TEST_CASE("8uP3", "[NPP.GeometricTransforms.Perspective]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    // NPP misses some pixels from the original ROI:
    cpu::Image<Pixel8uC1> cpu_mask     = cpu::Image<Pixel8uC1>::Load(root / "knownErrorsNPPperspective.tif");
    cpu::Image<Pixel8uC1> cpu_maskBack = cpu::Image<Pixel8uC1>::Load(root / "knownErrorsNPPperspectiveBack.tif");
    cpu::Image<Pixel8uC3> cpu_src1     = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");

    cpu_mask.SubInv(255);
    cpu_maskBack.SubInv(255);

    const int size = cpu_src1.Width();
    cpu::Image<Pixel8uC1> cpu_srcC1(size, size);
    cpu::Image<Pixel8uC1> cpu_srcC2(size, size);
    cpu::Image<Pixel8uC1> cpu_srcC3(size, size);

    cpu_src1.Copy(cpu_srcC1, cpu_srcC2, cpu_srcC3);

    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> cpu_dst2(size, size);

    cpu::Image<Pixel8uC1> cpu_dstC1(size, size);
    cpu::Image<Pixel8uC1> cpu_dstC2(size, size);
    cpu::Image<Pixel8uC1> cpu_dstC3(size, size);

    cpu::Image<Pixel8uC3> npp_res(size, size);
    cpu::Image<Pixel8uC3> npp_res2(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);
    nv::Image8uC3 npp_dst2(size, size);

    cpu_src1 >> npp_src1;

    AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation(Vec2d(-size / 2));
    AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
    AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation(Vec2d(size / 2));
    AffineTransformation<double> affine = shift2 * rot * shift1;
    PerspectiveTransformation<double> perspective(affine);
    perspective(2, 0) = 0.0002;
    perspective(2, 1) = -0.0003;

    // NearestNeighbor interpolation
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_dst2.Set({0, 128, 0}, nppCtx);
    npp_src1.WarpPerspective(npp_dst, perspective, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_dst.WarpPerspectiveBack(npp_dst2, perspective, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;
    npp_res2 << npp_dst2;

    cpu_dstC1.Set(128);
    cpu_dstC2.Set(0);
    cpu_dstC3.Set(0);

    cpu::Image<Pixel8uC3>::WarpPerspective(cpu_srcC1, cpu_srcC2, cpu_srcC3, cpu_dstC1, cpu_dstC2, cpu_dstC3,
                                           perspective, InterpolationMode::NearestNeighbor, BorderType::None);
    cpu::Image<Pixel8uC3>::Copy(cpu_dstC1, cpu_dstC2, cpu_dstC3, cpu_dst);

    cpu_srcC1.Set(0);
    cpu_srcC2.Set(128);
    cpu_srcC3.Set(0);

    cpu::Image<Pixel8uC3>::WarpPerspectiveBack(cpu_dstC1, cpu_dstC2, cpu_dstC3, cpu_srcC1, cpu_srcC2, cpu_srcC3,
                                               perspective, InterpolationMode::NearestNeighbor, BorderType::None);

    cpu::Image<Pixel8uC3>::Copy(cpu_srcC1, cpu_srcC2, cpu_srcC3, cpu_dst2);

    Vec3d summedErr;
    double summedErrChannel;
    cpu_dst.NormDiffL1Masked(npp_res, summedErr, summedErrChannel, cpu_mask);

    CHECK(summedErr == Vec3d{6, 3, 0}); // one pixel differs a bit

    cpu_dst2.NormDiffL1Masked(npp_res2, summedErr, summedErrChannel, cpu_maskBack);

    CHECK(summedErr == Vec3d{0, 0, 0});
}