#include "pixel32sC1.h"
#include "arithmetic.h"
#include "filtering.h"
#include "geometricTransforms.h"
#include "statistics.h"
#include "testBase.h"
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>   //NOLINT
#include <backends/simple_cpu/image/image.h>     //NOLINT
#include <backends/simple_cpu/image/imageView.h> //NOLINT
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

void runPixel32sC1(size_t aIterations, size_t aRepeats, int aWidth, int aHeight, const opp::image::Border &aBorder,
                   std::ofstream &aCsv)
{
    using oppT       = Pixel32sC1;
    using nppT       = nv::Image32sC1;
    using resFloatT  = Pixel32fC1;
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

    AddTest<oppT, nppT> testAdd(aIterations, aRepeats, aWidth, aHeight);
    SubTest<oppT, nppT> testSub(aIterations, aRepeats, aWidth, aHeight);
    MulTest<oppT, nppT> testMul(aIterations, aRepeats, aWidth, aHeight);
    DivTest<oppT, nppT> testDiv(aIterations, aRepeats, aWidth, aHeight);

    AffineTransformTest<oppT, nppT> testAffineTransformation(aIterations, aRepeats, aWidth, aHeight);

    PerspectiveTransformTest<oppT, nppT> testPerspectiveTransformation(aIterations, aRepeats, aWidth, aHeight);

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
    aCsv << testPerspectiveTransformation.GetResult<cpu::Image<oppT>, resDoubleT>();
}