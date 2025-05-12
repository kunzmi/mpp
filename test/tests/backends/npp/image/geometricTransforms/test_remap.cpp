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
#include <common/opp_defs.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC3", "[NPP.GeometricTransforms.Remap]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> cpu_dst2(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    cpu::Image<Pixel32fC2> cpu_coords(size, size);
    cpu::Image<Pixel32fC1> cpu_coordsX(size, size);
    cpu::Image<Pixel32fC1> cpu_coordsY(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);
    nv::Image32fC1 npp_coordsX(size, size);
    nv::Image32fC1 npp_coordsY(size, size);

    FillRandom<Pixel32fC2> op(seed);
    Pixel32fC2 randomVal;

    for (const auto &pixel : cpu_coords.SizeRoi())
    {
        op(randomVal);
        randomVal *= 4;
        randomVal -= 2; // range from -2..2
        randomVal += Vec2f(pixel.Pixel);
        cpu_coords(pixel.Pixel.x, pixel.Pixel.y)  = randomVal;
        cpu_coordsX(pixel.Pixel.x, pixel.Pixel.y) = randomVal.x;
        cpu_coordsY(pixel.Pixel.x, pixel.Pixel.y) = randomVal.y;
    }

    Vec3d maxErr;
    Vec3d summedErr;
    double summedErrChannel;

    cpu_src1 >> npp_src1;
    cpu_coordsX >> npp_coordsX;
    cpu_coordsY >> npp_coordsY;

    // Nearest neighbor
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_dst2.Set({128, 0, 0});
    cpu_src1.Remap(cpu_dst, cpu_coords, InterpolationMode::NearestNeighbor, BorderType::None);
    cpu_src1.Remap(cpu_dst2, cpu_coordsX, cpu_coordsY, InterpolationMode::NearestNeighbor, BorderType::None);

    // NPP samples also pixels outside the ROI when coordinate is > size-0.5, restrict it for comparison:
    cpu_dst.SetRoi(Border(0, 0, -2, -2));
    npp_res.SetRoi(Border(0, 0, -2, -2));

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);
    CHECK(summedErr == 0);

    cpu_dst.ResetRoi();
    cpu_dst2.NormDiffL1(cpu_dst, summedErr, summedErrChannel);
    CHECK(summedErr == 0);

    // linear
    cpu_dst.ResetRoi();
    npp_res.ResetRoi();
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_LINEAR), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Remap(cpu_dst, cpu_coords, InterpolationMode::Linear, BorderType::None);

    // NPP samples also pixels outside the ROI when coordinate is > size-0.5, restrict it for comparison:
    cpu_dst.SetRoi(Border(0, 0, -2, -2));
    npp_res.SetRoi(Border(0, 0, -2, -2));

    cpu_dst.NormDiffInf(npp_res, maxErr, summedErrChannel);
    CHECK(maxErr < 2); // max error is 1 due to a few rounding error...

    // cubic
    cpu_dst.ResetRoi();
    npp_res.ResetRoi();
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_CUBIC), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Remap(cpu_dst, cpu_coords, InterpolationMode::CubicLagrange, BorderType::None);

    // Interpolation at the border differs when outside ROI, restrict ROI:
    cpu_dst.SetRoi(Border(-2, -2, -4, -4));
    npp_res.SetRoi(Border(-2, -2, -4, -4));

    cpu_dst.NormDiffInf(npp_res, maxErr, summedErrChannel);
    CHECK(maxErr < 2); // max error is 1 due to a few rounding error...

    // CatmullRom
    cpu_dst.ResetRoi();
    npp_res.ResetRoi();
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_CUBIC2P_CATMULLROM), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Remap(cpu_dst, cpu_coords, InterpolationMode::Cubic2ParamCatmullRom, BorderType::None);

    // Interpolation at the border differs when outside ROI, restrict ROI:
    cpu_dst.SetRoi(Border(-3, -3, -5, -5));
    npp_res.SetRoi(Border(-3, -3, -5, -5));

    cpu_dst.NormDiffInf(npp_res, maxErr, summedErrChannel);
    CHECK(maxErr < 2); // max error is 1 due to a few rounding error...

    // B-spline
    cpu_dst.ResetRoi();
    npp_res.ResetRoi();
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_CUBIC2P_BSPLINE), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Remap(cpu_dst, cpu_coords, InterpolationMode::Cubic2ParamBSpline, BorderType::None);

    // Interpolation at the border differs when outside ROI, restrict ROI:
    cpu_dst.SetRoi(Border(-3, -3, -5, -5));
    npp_res.SetRoi(Border(-3, -3, -5, -5));

    cpu_dst.NormDiffInf(npp_res, maxErr, summedErrChannel);
    CHECK(maxErr < 2); // max error is 1 due to a few rounding error...

    // B05C03
    cpu_dst.ResetRoi();
    npp_res.ResetRoi();
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_CUBIC2P_B05C03), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Remap(cpu_dst, cpu_coords, InterpolationMode::Cubic2ParamB05C03, BorderType::None);

    // Interpolation at the border differs when outside ROI, restrict ROI:
    cpu_dst.SetRoi(Border(-3, -3, -5, -5));
    npp_res.SetRoi(Border(-3, -3, -5, -5));

    cpu_dst.NormDiffInf(npp_res, maxErr, summedErrChannel);
    CHECK(maxErr < 2); // max error is 1 due to a few rounding error...

    // Lanczos
    cpu_dst.ResetRoi();
    npp_res.ResetRoi();
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_LANCZOS), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Remap(cpu_dst, cpu_coords, InterpolationMode::Lanczos3Lobed, BorderType::None);

    // Interpolation at the border differs when outside ROI, restrict ROI:
    cpu_dst.SetRoi(Border(-5, -5, -7, -7));
    npp_res.SetRoi(Border(-5, -5, -7, -7));

    cpu_dst.NormDiffInf(npp_res, maxErr, summedErrChannel);
    CHECK(maxErr < 2); // max error is 1 due to a few rounding error...
}

TEST_CASE("8uP3", "[NPP.GeometricTransforms.Remap]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> cpu_dst2(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    cpu::Image<Pixel32fC2> cpu_coords(size, size);
    cpu::Image<Pixel32fC1> cpu_coordsX(size, size);
    cpu::Image<Pixel32fC1> cpu_coordsY(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);
    nv::Image32fC1 npp_coordsX(size, size);
    nv::Image32fC1 npp_coordsY(size, size);

    cpu::Image<Pixel8uC1> cpu_srcC1(size, size);
    cpu::Image<Pixel8uC1> cpu_srcC2(size, size);
    cpu::Image<Pixel8uC1> cpu_srcC3(size, size);

    cpu::Image<Pixel8uC1> cpu_dstC1(size, size);
    cpu::Image<Pixel8uC1> cpu_dstC2(size, size);
    cpu::Image<Pixel8uC1> cpu_dstC3(size, size);

    cpu_src1.Copy(cpu_srcC1, cpu_srcC2, cpu_srcC3);

    FillRandom<Pixel32fC2> op(seed);
    Pixel32fC2 randomVal;

    for (const auto &pixel : cpu_coords.SizeRoi())
    {
        op(randomVal);
        randomVal *= 4;
        randomVal -= 2; // range from -2..2
        randomVal += Vec2f(pixel.Pixel);
        cpu_coords(pixel.Pixel.x, pixel.Pixel.y)  = randomVal;
        cpu_coordsX(pixel.Pixel.x, pixel.Pixel.y) = randomVal.x;
        cpu_coordsY(pixel.Pixel.x, pixel.Pixel.y) = randomVal.y;
    }

    Vec3d summedErr;
    double summedErrChannel;

    cpu_src1 >> npp_src1;
    cpu_coordsX >> npp_coordsX;
    cpu_coordsY >> npp_coordsY;

    // Nearest neighbor
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Remap(npp_coordsX, npp_coordsY, npp_dst, static_cast<int>(NPPI_INTER_NN), nppCtx);
    npp_res << npp_dst;

    cpu_dstC1.Set(128);
    cpu_dstC2.Set(0);
    cpu_dstC3.Set(0);
    cpu::Image<Pixel8uC3>::Remap(cpu_srcC1, cpu_srcC2, cpu_srcC3, cpu_dstC1, cpu_dstC2, cpu_dstC3, cpu_coords,
                                 InterpolationMode::NearestNeighbor, BorderType::None);
    cpu::Image<Pixel8uC3>::Copy(cpu_dstC1, cpu_dstC2, cpu_dstC3, cpu_dst);

    cpu_dstC1.Set(128);
    cpu_dstC2.Set(0);
    cpu_dstC3.Set(0);
    cpu::Image<Pixel8uC3>::Remap(cpu_srcC1, cpu_srcC2, cpu_srcC3, cpu_dstC1, cpu_dstC2, cpu_dstC3, cpu_coordsX,
                                 cpu_coordsY, InterpolationMode::NearestNeighbor, BorderType::None);
    cpu::Image<Pixel8uC3>::Copy(cpu_dstC1, cpu_dstC2, cpu_dstC3, cpu_dst2);

    // NPP samples also pixels outside the ROI when coordinate is > size-0.5, restrict it for comparison:
    cpu_dst.SetRoi(Border(0, 0, -2, -2));
    npp_res.SetRoi(Border(0, 0, -2, -2));

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);
    CHECK(summedErr == 0);

    cpu_dst.ResetRoi();
    cpu_dst2.NormDiffL1(cpu_dst, summedErr, summedErrChannel);
    CHECK(summedErr == 0);
}