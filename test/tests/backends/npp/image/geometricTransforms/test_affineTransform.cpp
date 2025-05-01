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

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 512;

TEST_CASE("8uC3", "[NPP.GeometricTransforms.Affine]")
{
    // const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load("Y:\\Images\\TestImages\\512.tif");
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1 >> npp_src1;

    AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation({-256, -256});
    AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
    AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation({256, 256});
    AffineTransformation<double> affine = shift2 * rot * shift1;
    AffineTransformation<double> affine2Npp;
    AffineTransformation<float> affinef(affine);
    npp_dst.Set({0, 0, 0}, nppCtx);
    npp_src1.WarpAffine(npp_dst, affine, static_cast<int>(NPPI_INTER_CUBIC), nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set({0, 0, 0});

    Quad<double> quadNpp;

    npp_src1.GetAffineQuad(quadNpp, affine);

    npp_src1.GetAffineTransform(quadNpp, affine2Npp);

    Quad<double> quad0 = affine * cpu_src1.ROI();

    Bound<double> bound(quad0);

    AffineTransformation<double> affine2Opp(cpu_src1.ROI(), quad0);

    std::cout << "NPP:" << std::endl;
    std::cout << quadNpp << std::endl << std::endl;

    std::cout << "OPP:" << std::endl;
    std::cout << quad0 << std::endl << std::endl;

    std::cout << "NPP:" << std::endl;
    std::cout << affine2Npp << std::endl << std::endl;

    std::cout << "OPP:" << std::endl;
    std::cout << affine2Opp << std::endl << std::endl;

    std::cout << "Orig:" << std::endl;
    std::cout << affine << std::endl << std::endl;

    std::cout << bound << std::endl;

    cpu_src1.WarpAffine(cpu_dst, affinef, InterpolationMode::CubicLagrange, BorderType::None);
    Vec3d maxErr;
    double maxErrChannel;
    cpu_dst.NormDiffL1(npp_res, maxErr, maxErrChannel);
    std::cout << maxErr << std::endl;

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
    npp_res.Save("Y:\\Images\\TestImages\\512rotNPP.tif");
    cpu_dst.Save("Y:\\Images\\TestImages\\512rotOPP.tif");
}