#include "arithmetic.h"
#include "geometricTransforms.h"
#include "pixel32sC1.h"
#include "testBase.h"
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>   //NOLINT
#include <backends/simple_cpu/image/image.h>     //NOLINT
#include <backends/simple_cpu/image/imageView.h> //NOLINT
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/mpp_defs.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <vector>

using namespace mpp;
using namespace mpp::cuda;
using namespace mpp::image;
using namespace mpp::image::cuda;
namespace nv  = mpp::image::npp;
namespace cpu = mpp::image::cpuSimple;

void runPixel32sC1(size_t aIterations, size_t aRepeats, int aWidth, int aHeight, const mpp::image::Border &aBorder,
                   std::vector<mpp::TestResult> &aTestResult, TestsToRun aTestsToRun)
{
    using mppT       = Pixel32sC1;
    using nppT       = nv::Image32sC1;
    using resDoubleT = Pixel64fC1;

    Roi roi(0, 0, aWidth, aHeight);
    roi += aBorder;

    const AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation(Vec2d(-aWidth / 2));
    const AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
    const AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation(Vec2d(aWidth / 2));
    const AffineTransformation<double> affine = shift2 * rot * shift1;
    PerspectiveTransformation<double> perspective(affine);
    perspective(2, 0) = 0.0002;
    perspective(2, 1) = -0.0003;

    AddTest<mppT, nppT> testAdd(aIterations, aRepeats, aWidth, aHeight);
    SubTest<mppT, nppT> testSub(aIterations, aRepeats, aWidth, aHeight);
    MulTest<mppT, nppT> testMul(aIterations, aRepeats, aWidth, aHeight);
    DivTest<mppT, nppT> testDiv(aIterations, aRepeats, aWidth, aHeight);

    AffineTransformTest<mppT, nppT> testAffineTransformation(aIterations, aRepeats, aWidth, aHeight);

    PerspectiveTransformTest<mppT, nppT> testPerspectiveTransformation(aIterations, aRepeats, aWidth, aHeight);

    cpu::Image<mppT> cpu_src1(aWidth, aHeight);
    cpu::Image<mppT> cpu_src2(aWidth, aHeight);
    cpu_src1.FillRandom(0);
    cpu_src2.FillRandom(1);
    cpu_src2.Add({1}); // avoid division by 0...

    if (aTestsToRun.Arithmetic)
    {
        testAdd.Init(cpu_src1, cpu_src2);
        testAdd.Run(roi);
        aTestResult.emplace_back(testAdd.GetResult<cpu::Image<mppT>, resDoubleT>());

        testSub.Init(cpu_src1, cpu_src2);
        testSub.Run(roi);
        aTestResult.emplace_back(testSub.GetResult<cpu::Image<mppT>, resDoubleT>());

        testMul.Init(cpu_src1, cpu_src2);
        testMul.Run(roi);
        aTestResult.emplace_back(testMul.GetResult<cpu::Image<mppT>, resDoubleT>());

        testDiv.Init(cpu_src1, cpu_src2);
        testDiv.Run(roi);
        aTestResult.emplace_back(testDiv.GetResult<cpu::Image<mppT>, resDoubleT>());
    }

    if (aTestsToRun.GeometryTransform)
    {
        testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::NearestNeighbor);
        testAffineTransformation.Run(roi);
        aTestResult.emplace_back(testAffineTransformation.GetResult<cpu::Image<mppT>, resDoubleT>());

        testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::Linear);
        testAffineTransformation.Run(roi);
        aTestResult.emplace_back(testAffineTransformation.GetResult<cpu::Image<mppT>, resDoubleT>());

        testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::CubicLagrange);
        testAffineTransformation.Run(roi);
        aTestResult.emplace_back(testAffineTransformation.GetResult<cpu::Image<mppT>, resDoubleT>());

        testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::NearestNeighbor);
        testPerspectiveTransformation.Run(roi);
        aTestResult.emplace_back(testPerspectiveTransformation.GetResult<cpu::Image<mppT>, resDoubleT>());

        testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::Linear);
        testPerspectiveTransformation.Run(roi);
        aTestResult.emplace_back(testPerspectiveTransformation.GetResult<cpu::Image<mppT>, resDoubleT>());

        testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::CubicLagrange);
        testPerspectiveTransformation.Run(roi);
        aTestResult.emplace_back(testPerspectiveTransformation.GetResult<cpu::Image<mppT>, resDoubleT>());
    }
}