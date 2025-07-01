#include "pixel16uC3.h"
#include "arithmetic.h"
#include "filtering.h"
#include "geometricTransforms.h"
#include "statistics.h"
#include "testBase.h"
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC3View.h> //NOLINT
#include <backends/npp/image/image32f.h>       //NOLINT
#include <backends/npp/image/image32fC1View.h> //NOLINT
#include <backends/simple_cpu/image/image.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/filterArea.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/mpp_defs.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <cstddef>
#include <vector>

using namespace mpp;
using namespace mpp::cuda;
using namespace mpp::image;
using namespace mpp::image::cuda;
namespace nv  = mpp::image::npp;
namespace cpu = mpp::image::cpuSimple;

void runPixel16uC3(size_t aIterations, size_t aRepeats, int aWidth, int aHeight, const mpp::image::Border &aBorder,
                   std::vector<mpp::TestResult> &aTestResult, TestsToRun aTestsToRun)
{
    using mppT       = Pixel16uC3;
    using nppT       = nv::Image16uC3;
    using resFloatT  = Pixel32fC3;
    using resDoubleT = Pixel64fC3;

    Roi roi(0, 0, aWidth, aHeight);
    roi += aBorder;

    const FilterArea filterAreaCol({1, 11}, {0, 5});
    const FilterArea filterAreaRow({11, 1}, {5, 0});
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

    BoxFilterTest<mppT, nppT> testBoxFilter(aIterations, aRepeats, aWidth, aHeight);
    RowWindowSumTest<mppT, resFloatT, nppT, nv::Image32fC3> testRowWindowSum(aIterations, aRepeats, aWidth, aHeight);
    ColumnWindowSumTest<mppT, resFloatT, nppT, nv::Image32fC3> testColumnWindowSum(aIterations, aRepeats, aWidth,
                                                                                   aHeight);
    LowPassFilterTest<mppT, nppT> testLowPassFilter(aIterations, aRepeats, aWidth, aHeight);
    GaussFilterTest<mppT, nppT> testGaussFilter(aIterations, aRepeats, aWidth, aHeight);
    GaussAdvancedFilterTest<mppT, nppT> testGaussAdvancedilter(aIterations, aRepeats, aWidth, aHeight);

    AffineTransformTest<mppT, nppT> testAffineTransformation(aIterations, aRepeats, aWidth, aHeight);

    PerspectiveTransformTest<mppT, nppT> testPerspectiveTransformation(aIterations, aRepeats, aWidth, aHeight);

    QualityIndexTest<mppT, resDoubleT, nppT, float, vector_active_size_v<mppT>> testQualityIndex(aIterations, aRepeats,
                                                                                                 aWidth, aHeight);

    MaxTest<mppT, mppT, nppT, mpp::ushort, vector_active_size_v<mppT>> testMax(aIterations, aRepeats, aWidth, aHeight);

    MeanTest<mppT, resDoubleT, nppT, double, vector_active_size_v<mppT>> testMean(aIterations, aRepeats, aWidth,
                                                                                  aHeight);

    MeanStdTest<mppT, resDoubleT, nppT, double, vector_active_size_v<mppT>> testMeanStd(aIterations, aRepeats, aWidth,
                                                                                        aHeight);

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

    if (aTestsToRun.Filtering)
    {
        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, 5);
        aTestResult.emplace_back(testBoxFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, 9);
        aTestResult.emplace_back(testBoxFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, 11);
        aTestResult.emplace_back(testBoxFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, 21);
        aTestResult.emplace_back(testBoxFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, 51);
        aTestResult.emplace_back(testBoxFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, 101);
        aTestResult.emplace_back(testBoxFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, 151);
        aTestResult.emplace_back(testBoxFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testRowWindowSum.Init(cpu_src1);
        testRowWindowSum.Run(roi, filterAreaRow);
        aTestResult.emplace_back(testRowWindowSum.GetResult<cpu::Image<resFloatT>, resDoubleT>());

        testColumnWindowSum.Init(cpu_src1);
        testColumnWindowSum.Run(roi, filterAreaCol);
        aTestResult.emplace_back(testColumnWindowSum.GetResult<cpu::Image<resFloatT>, resDoubleT>());

        testLowPassFilter.Init(cpu_src1);
        testLowPassFilter.Run(roi, 3);
        aTestResult.emplace_back(testLowPassFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testLowPassFilter.Init(cpu_src1);
        testLowPassFilter.Run(roi, 5);
        aTestResult.emplace_back(testLowPassFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 3);
        aTestResult.emplace_back(testGaussFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 5);
        aTestResult.emplace_back(testGaussFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 7);
        aTestResult.emplace_back(testGaussFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 9);
        aTestResult.emplace_back(testGaussFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 11);
        aTestResult.emplace_back(testGaussFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 13);
        aTestResult.emplace_back(testGaussFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 15);
        aTestResult.emplace_back(testGaussFilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussAdvancedilter.Init(cpu_src1, 7);
        testGaussAdvancedilter.Run(roi, 7);
        aTestResult.emplace_back(testGaussAdvancedilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussAdvancedilter.Init(cpu_src1, 9);
        testGaussAdvancedilter.Run(roi, 9);
        aTestResult.emplace_back(testGaussAdvancedilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussAdvancedilter.Init(cpu_src1, 11);
        testGaussAdvancedilter.Run(roi, 11);
        aTestResult.emplace_back(testGaussAdvancedilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussAdvancedilter.Init(cpu_src1, 21);
        testGaussAdvancedilter.Run(roi, 21);
        aTestResult.emplace_back(testGaussAdvancedilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussAdvancedilter.Init(cpu_src1, 51);
        testGaussAdvancedilter.Run(roi, 51);
        aTestResult.emplace_back(testGaussAdvancedilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussAdvancedilter.Init(cpu_src1, 101);
        testGaussAdvancedilter.Run(roi, 101);
        aTestResult.emplace_back(testGaussAdvancedilter.GetResult<cpu::Image<mppT>, resDoubleT>());

        testGaussAdvancedilter.Init(cpu_src1, 151);
        testGaussAdvancedilter.Run(roi, 151);
        aTestResult.emplace_back(testGaussAdvancedilter.GetResult<cpu::Image<mppT>, resDoubleT>());
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

    if (aTestsToRun.Statistics)
    {
        testQualityIndex.Init(cpu_src1, cpu_src2);
        testQualityIndex.Run(roi);
        aTestResult.emplace_back(testQualityIndex.GetResult<resDoubleT, resFloatT>());

        testMax.Init(cpu_src1);
        testMax.Run(roi);
        aTestResult.emplace_back(testMax.GetResult<mppT, mppT>());

        testMean.Init(cpu_src1);
        testMean.Run(roi);
        aTestResult.emplace_back(testMean.GetResult<resDoubleT, resDoubleT>());

        testMeanStd.Init(cpu_src1);
        testMeanStd.Run(roi);
        aTestResult.emplace_back(testMeanStd.GetResult<resDoubleT, resDoubleT>());
    }
}