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

TEST_CASE("8uC3", "[NPP.GeometricTransforms.Mirror]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    Vec3d summedErr;
    double summedErrChannel;

    cpu_src1 >> npp_src1;

    // Horizontal
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Mirror(npp_dst, NppiAxis::NPP_HORIZONTAL_AXIS, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Mirror(cpu_dst, MirrorAxis::Horizontal);

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // Vertical
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Mirror(npp_dst, NppiAxis::NPP_VERTICAL_AXIS, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Mirror(cpu_dst, MirrorAxis::Vertical);

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // Both
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Mirror(npp_dst, NppiAxis::NPP_BOTH_AXIS, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Mirror(cpu_dst, MirrorAxis::Both);

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // Horizontal inplace
    npp_src1.Mirror(NppiAxis::NPP_HORIZONTAL_AXIS, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Mirror(MirrorAxis::Horizontal);

    cpu_src1.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // Horizontal inplace
    npp_src1.Mirror(NppiAxis::NPP_VERTICAL_AXIS, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Mirror(MirrorAxis::Vertical);

    cpu_src1.NormDiffL1(npp_res, summedErr, summedErrChannel);
    CHECK(summedErr == 0);

    // Both inplace
    npp_src1.Mirror(NppiAxis::NPP_BOTH_AXIS, nppCtx);
    npp_res << npp_src1;

    cpu_src1.Mirror(MirrorAxis::Both);

    cpu_src1.NormDiffL1(npp_res, summedErr, summedErrChannel);

    cpu_src1.Save(root / "mirrorOpp.tif");
    CHECK(summedErr == 0);
}

TEST_CASE("8uC3 - Uneven", "[NPP.GeometricTransforms.Mirror]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    std::filesystem::path root    = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC3> cpu_src = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");

    cpu_src.SetRoi(Roi(0, 0, size - 1, size - 1));

    cpu::Image<Pixel8uC3> cpu_src1(size - 1, size - 1);
    cpu::Image<Pixel8uC3> cpu_dst(size - 1, size - 1);
    cpu::Image<Pixel8uC3> npp_res(size - 1, size - 1);
    nv::Image8uC3 npp_src1(size - 1, size - 1);
    nv::Image8uC3 npp_dst(size - 1, size - 1);

    cpu_src.Copy(cpu_src1);
    Vec3d summedErr;
    double summedErrChannel;

    cpu_src1 >> npp_src1;

    // Horizontal
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Mirror(npp_dst, NppiAxis::NPP_HORIZONTAL_AXIS, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Mirror(cpu_dst, MirrorAxis::Horizontal);

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // Vertical
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Mirror(npp_dst, NppiAxis::NPP_VERTICAL_AXIS, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Mirror(cpu_dst, MirrorAxis::Vertical);

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // Both
    npp_dst.Set({128, 0, 0}, nppCtx);
    npp_src1.Mirror(npp_dst, NppiAxis::NPP_BOTH_AXIS, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({128, 0, 0});
    cpu_src1.Mirror(cpu_dst, MirrorAxis::Both);

    cpu_dst.NormDiffL1(npp_res, summedErr, summedErrChannel);

    CHECK(summedErr == 0);

    // Horizontal inplace
    // NPP doesn't support uneven inplace...
    cpu_src1.Mirror(MirrorAxis::Horizontal);

    // Horizontal inplace
    cpu_src1.Mirror(MirrorAxis::Vertical);

    // Both inplace
    cpu_src1.Mirror(MirrorAxis::Both);

    // should be again same as original:
    cpu_src1.NormDiffL1(cpu_src, summedErr, summedErrChannel);

    CHECK(summedErr == 0);
}