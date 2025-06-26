#include "pixel8uC1.h"
#include "arithmetic.h"
#include "filtering.h"
#include "geometricTransforms.h"
#include "statistics.h"
#include "testBase.h"
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h> //NOLINT
#include <backends/npp/image/image8u.h>        //NOLINT
#include <backends/npp/image/image8uC1View.h>  //NOLINT
#include <backends/simple_cpu/image/image.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/filterArea.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/opp_defs.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <cstddef>
#include <fstream>

using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;
namespace nv  = opp::image::npp;
namespace cpu = opp::image::cpuSimple;

void runPixel8uC1(size_t aIterations, size_t aRepeats, int aWidth, int aHeight, const opp::image::Border &aBorder,
                  std::ofstream &aCsv)
{
    using oppT       = Pixel8uC1;
    using nppT       = nv::Image8uC1;
    using resFloatT  = Pixel32fC1;
    using resDoubleT = Pixel64fC1;

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

    AddTest<oppT, nppT> testAdd(aIterations, aRepeats, aWidth, aHeight);
    SubTest<oppT, nppT> testSub(aIterations, aRepeats, aWidth, aHeight);
    MulTest<oppT, nppT> testMul(aIterations, aRepeats, aWidth, aHeight);
    DivTest<oppT, nppT> testDiv(aIterations, aRepeats, aWidth, aHeight);

    BoxFilterTest<oppT, nppT> testBoxFilter(aIterations, aRepeats, aWidth, aHeight);
    RowWindowSumTest<oppT, resFloatT, nppT, nv::Image32fC1> testRowWindowSum(aIterations, aRepeats, aWidth, aHeight);
    ColumnWindowSumTest<oppT, resFloatT, nppT, nv::Image32fC1> testColumnWindowSum(aIterations, aRepeats, aWidth,
                                                                                   aHeight);
    LowPassFilterTest<oppT, nppT> testLowPassFilter(aIterations, aRepeats, aWidth, aHeight);
    GaussFilterTest<oppT, nppT> testGaussFilter(aIterations, aRepeats, aWidth, aHeight);
    GaussAdvancedFilterTest<oppT, nppT> testGaussAdvancedilter(aIterations, aRepeats, aWidth, aHeight);

    AffineTransformTest<oppT, nppT> testAffineTransformation(aIterations, aRepeats, aWidth, aHeight);

    PerspectiveTransformTest<oppT, nppT> testPerspectiveTransformation(aIterations, aRepeats, aWidth, aHeight);

    QualityIndexTest<oppT, resDoubleT, nppT, float, vector_active_size_v<oppT>> testQualityIndex(aIterations, aRepeats,
                                                                                                 aWidth, aHeight);

    MaxTest<oppT, oppT, nppT, byte, vector_active_size_v<oppT>> testMax(aIterations, aRepeats, aWidth, aHeight);

    MeanTest<oppT, resDoubleT, nppT, double, vector_active_size_v<oppT>> testMean(aIterations, aRepeats, aWidth,
                                                                                  aHeight);

    MeanStdTest<oppT, resDoubleT, nppT, double, vector_active_size_v<oppT>> testMeanStd(aIterations, aRepeats, aWidth,
                                                                                        aHeight);

    MSETest<oppT, resDoubleT, nppT, float, vector_active_size_v<oppT>> testMSE(aIterations, aRepeats, aWidth, aHeight);

    cpu::Image<oppT> cpu_src1(aWidth, aHeight);
    cpu::Image<oppT> cpu_src2(aWidth, aHeight);
    cpu_src1.FillRandom(0);
    cpu_src2.FillRandom(1);
    cpu_src2.Add({1}); // avoid division by 0...

    testAdd.Init(cpu_src1, cpu_src2);
    testAdd.Run(roi);
    aCsv << testAdd.GetResult<cpu::Image<oppT>, resDoubleT>();

    testSub.Init(cpu_src1, cpu_src2);
    testSub.Run(roi);
    aCsv << testSub.GetResult<cpu::Image<oppT>, resDoubleT>();

    testMul.Init(cpu_src1, cpu_src2);
    testMul.Run(roi);
    aCsv << testMul.GetResult<cpu::Image<oppT>, resDoubleT>();

    testDiv.Init(cpu_src1, cpu_src2);
    testDiv.Run(roi);
    aCsv << testDiv.GetResult<cpu::Image<oppT>, resDoubleT>();

    /*testBoxFilter.Init(cpu_src1);
    testBoxFilter.Run(roi, 5);
    aCsv << testBoxFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testBoxFilter.Init(cpu_src1);
    testBoxFilter.Run(roi, 9);
    aCsv << testBoxFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testBoxFilter.Init(cpu_src1);
    testBoxFilter.Run(roi, 11);
    aCsv << testBoxFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testBoxFilter.Init(cpu_src1);
    testBoxFilter.Run(roi, 21);
    aCsv << testBoxFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testBoxFilter.Init(cpu_src1);
    testBoxFilter.Run(roi, 51);
    aCsv << testBoxFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testBoxFilter.Init(cpu_src1);
    testBoxFilter.Run(roi, 101);
    aCsv << testBoxFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testBoxFilter.Init(cpu_src1);
    testBoxFilter.Run(roi, 151);
    aCsv << testBoxFilter.GetResult<cpu::Image<oppT>, resDoubleT>();*/

    /*testRowWindowSum.Init(cpu_src1);
    testRowWindowSum.Run(roi, filterAreaRow);
    aCsv << testRowWindowSum.GetResult<cpu::Image<resFloatT>, resDoubleT>();

    testColumnWindowSum.Init(cpu_src1);
    testColumnWindowSum.Run(roi, filterAreaCol);
    aCsv << testColumnWindowSum.GetResult<cpu::Image<resFloatT>, resDoubleT>();

    testLowPassFilter.Init(cpu_src1);
    testLowPassFilter.Run(roi, 3);
    aCsv << testLowPassFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testLowPassFilter.Init(cpu_src1);
    testLowPassFilter.Run(roi, 5);
    aCsv << testLowPassFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussFilter.Init(cpu_src1);
    testGaussFilter.Run(roi, 3);
    aCsv << testGaussFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussFilter.Init(cpu_src1);
    testGaussFilter.Run(roi, 5);
    aCsv << testGaussFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussFilter.Init(cpu_src1);
    testGaussFilter.Run(roi, 7);
    aCsv << testGaussFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussFilter.Init(cpu_src1);
    testGaussFilter.Run(roi, 9);
    aCsv << testGaussFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussFilter.Init(cpu_src1);
    testGaussFilter.Run(roi, 11);
    aCsv << testGaussFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussFilter.Init(cpu_src1);
    testGaussFilter.Run(roi, 13);
    aCsv << testGaussFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussFilter.Init(cpu_src1);
    testGaussFilter.Run(roi, 15);
    aCsv << testGaussFilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussAdvancedilter.Init(cpu_src1, 7);
    testGaussAdvancedilter.Run(roi, 7);
    aCsv << testGaussAdvancedilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussAdvancedilter.Init(cpu_src1, 9);
    testGaussAdvancedilter.Run(roi, 9);
    aCsv << testGaussAdvancedilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussAdvancedilter.Init(cpu_src1, 11);
    testGaussAdvancedilter.Run(roi, 11);
    aCsv << testGaussAdvancedilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussAdvancedilter.Init(cpu_src1, 21);
    testGaussAdvancedilter.Run(roi, 21);
    aCsv << testGaussAdvancedilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussAdvancedilter.Init(cpu_src1, 51);
    testGaussAdvancedilter.Run(roi, 51);
    aCsv << testGaussAdvancedilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussAdvancedilter.Init(cpu_src1, 101);
    testGaussAdvancedilter.Run(roi, 101);
    aCsv << testGaussAdvancedilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testGaussAdvancedilter.Init(cpu_src1, 151);
    testGaussAdvancedilter.Run(roi, 151);
    aCsv << testGaussAdvancedilter.GetResult<cpu::Image<oppT>, resDoubleT>();

    testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::NearestNeighbor);
    testAffineTransformation.Run(roi);
    aCsv << testAffineTransformation.GetResult<cpu::Image<oppT>, resDoubleT>();

    testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::Linear);
    testAffineTransformation.Run(roi);
    aCsv << testAffineTransformation.GetResult<cpu::Image<oppT>, resDoubleT>();

    testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::CubicLagrange);
    testAffineTransformation.Run(roi);
    aCsv << testAffineTransformation.GetResult<cpu::Image<oppT>, resDoubleT>();

    testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::NearestNeighbor);
    testPerspectiveTransformation.Run(roi);
    aCsv << testPerspectiveTransformation.GetResult<cpu::Image<oppT>, resDoubleT>();

    testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::Linear);
    testPerspectiveTransformation.Run(roi);
    aCsv << testPerspectiveTransformation.GetResult<cpu::Image<oppT>, resDoubleT>();

    testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::CubicLagrange);
    testPerspectiveTransformation.Run(roi);
    aCsv << testPerspectiveTransformation.GetResult<cpu::Image<oppT>, resDoubleT>();*/

    testQualityIndex.Init(cpu_src1, cpu_src2);
    testQualityIndex.Run(roi);
    aCsv << testQualityIndex.GetResult<resDoubleT, resFloatT>();

    testMax.Init(cpu_src1);
    testMax.Run(roi);
    aCsv << testMax.GetResult<oppT, oppT>();

    testMean.Init(cpu_src1);
    testMean.Run(roi);
    aCsv << testMean.GetResult<resDoubleT, resDoubleT>();

    testMeanStd.Init(cpu_src1);
    testMeanStd.Run(roi);
    aCsv << testMeanStd.GetResult<resDoubleT, resDoubleT>();

    testMSE.Init(cpu_src1, cpu_src2);
    testMSE.Run(roi);
    aCsv << testMSE.GetResult<resDoubleT, resFloatT>();
}