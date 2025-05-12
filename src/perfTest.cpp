#include "backends/npp/image/imageView.h"
#include "common/exception.h"
#include "common/image/pixelTypes.h"
#include <algorithm>
#include <backends/cuda/devVar.h>     // NOLINT(misc-include-cleaner)
#include <backends/cuda/devVarView.h> // NOLINT(misc-include-cleaner)
#include <backends/cuda/event.h>
#include <backends/cuda/image/image.h>     // NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView.h> // NOLINT(misc-include-cleaner)
#include <backends/cuda/image/statistics/integralSqr.h>
#include <backends/cuda/image/statistics/ssim.h>
#include <backends/cuda/stream.h>
#include <backends/cuda/streamCtx.h>
#include <backends/npp/image/image16u.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image16uC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32f.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32fC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32s.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image8u.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image8uC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <cmath>
#include <common/image/functors/reductionInitValues.h>
#include <common/scratchBuffer.h>
#include <common/version.h>
#include <cstddef>
#include <filesystem> // NOLINT(misc-include-cleaner)
#include <ios>
#include <iostream>
#include <nppcore.h>
#include <nppdefs.h>
#include <numeric>
#include <vector>

#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>

using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;
namespace nv  = opp::image::npp;
namespace cpu = opp::image::cpuSimple;

int main()
{
    try
    {
        // FixedFilterKernel<opp::FixedFilter::HighPass, 3, int>::Values[0][0];

        constexpr size_t iterations = 10;
        constexpr size_t repeats    = 100; /**/
        constexpr int imgWidth      = 4096 * 1;
        constexpr int imgHeight     = 4096 * 1;
        // INT_MAX / 1024 / 256 + 1;

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        cpu::Image<Pixel32fC1> cpu_src1(imgWidth, imgHeight);
        // cpu::ImageView<Pixel8u3> cpu_src1A(cpu_src1);
        /*cpu::Image<Pixel8uC1> cpu_src1 =
            cpu::Image<Pixel8uC1>::Load("C:\\Users\\kunz_\\source\\repos\\oppV1\\opp\\test\\testData\\bird256bw.tif");*/
        cpu::Image<Pixel32fC1> cpu_dst(imgWidth, imgHeight);

        Image<Pixel32fC1> opp_src1(imgWidth, imgHeight);
        Image<Pixel32fC1> opp_dst(imgWidth, imgHeight);

        nv::Image32fC1 npp_src1(imgWidth, imgHeight);
        nv::Image32fC1 npp_dst(imgWidth, imgHeight);

        DevVar<float> filter(15 * 15);

        /*for (auto &pixel : cpu_src1)
        {
            pixel.Value().x = to_float(pixel.Pixel().x % 256);
        } */

        // cpu_src1.Set(1);
        //  cpu_src1.FillRandom(0);
        //   cpu_src2.FillRandom(1);
        //   cpu_src2.Div(4).Add(cpu_src1).Sub(2); /**/
        /*cpu_mask.FillRandom(2);
        cpu_mask.Div(127).Mul(255);*/
        /*cpu_src1.Set(1);
        cpu_mask.Set(1);
        cpu_src1.Mul(0.05f);
        cpu_src1.Add(cpu_src2);*/

        cpu_src1 >> opp_src1;
        // cpu_src2 >> opp_src2;
        //  cpu_mask >> opp_mask;

        cpu_src1 >> npp_src1;
        // cpu_src2 >> npp_src2;
        //  cpu_mask >> npp_mask;

        NppStreamContext nppCtx;
        NppStatus status = nppGetStreamContext(&nppCtx);
        if (status != NPP_SUCCESS)
        {
            return -1;
        }

        StreamCtx oppCtx = StreamCtxSingleton::Get();

        // opp_src1.SetRoi(Border(-1, 0));
        /*opp_src1.SetRoi(-1);
        opp_dst.SetRoi(-1);

        npp_src1.SetRoi(-1);
        npp_dst.SetRoi(-1); */
        const BorderType opp_boder = BorderType::Replicate;

        // npp_src1.FilterBorder32f(npp_dst, filter, {15, 15}, {6, 6}, NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);

        opp_src1.FixedFilter(opp_dst, FixedFilter::Gauss, MaskSize::Mask_15x15, opp_boder, {0}, Roi(), oppCtx);

        npp_src1.FilterGaussBorder(npp_dst, NppiMaskSize::NPP_MASK_SIZE_15_X_15, NppiBorderType::NPP_BORDER_REPLICATE,
                                   nppCtx);

        /*cpu_src1.Save("f:\\highpassOrig.tif");
        cpu_dst << opp_dst;
        cpu_dst.Save("f:\\highpassOpp.tif");
        cpu_dst << npp_dst;
        cpu_dst.Save("f:\\highpassNpp.tif"); */

        std::vector<float> runtimeOPP(iterations);
        std::vector<float> runtimeNPP(iterations);

        const Event startGlobalOPP;
        const Event startIterOPP;
        const Event endIterOPP;
        const Event endGlobalOPP;
        const Event startGlobalNPP;
        const Event startIterNPP;
        const Event endIterNPP;
        const Event endGlobalNPP;

        startGlobalOPP.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIterOPP.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                opp_src1.FixedFilter(opp_dst, FixedFilter::Gauss, MaskSize::Mask_15x15, opp_boder, {0}, Roi(), oppCtx);
            }
            endIterOPP.Record();
            endIterOPP.Synchronize();
            const float runtime = endIterOPP - startIterOPP;
            runtimeOPP[i]       = runtime;
        }
        endGlobalOPP.Record();
        endGlobalOPP.Synchronize();
        const float runtimeGlobalOPP = endGlobalOPP - startGlobalOPP;

        startGlobalNPP.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIterNPP.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                npp_src1.FilterGaussBorder(npp_dst, NppiMaskSize::NPP_MASK_SIZE_15_X_15,
                                           NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);
            }
            endIterNPP.Record();
            endIterNPP.Synchronize();
            const float runtime = endIterNPP - startIterNPP;
            runtimeNPP[i]       = runtime;
        }
        endGlobalNPP.Record();
        endGlobalNPP.Synchronize();
        const float runtimeGlobalNPP = endGlobalNPP - startGlobalNPP;

        // Pixel64fC1 h_resOPPQI;
        // resOppQI >> h_resOPPQI;
        ///*Pixel64fC1 h_resOPPStd;
        // resOppStd >> h_resOPPStd;*/
        ///*double h_resOPPScalarMean;
        // resOppScalarMean >> h_resOPPScalarMean;
        // double h_resOPPScalarStd;
        // resOppScalarStd >> h_resOPPScalarStd;*/
        // Pixel32fC1 h_resNPPMean;
        // resNpp1 >> h_resNPPMean.data();
        ///*Pixel64fC1 h_resNPPStd;
        // resNpp2 >> h_resNPPStd.data();*/

        ///*std::vector<Pixel32fC3> h_buffer(imgHeight);
        // bufferOppMin >> h_buffer;

        // for (const auto &elem : h_buffer)
        //{
        //     std::cout << elem << std::endl;
        // }*/

        // std::cout << "Result OPP: " << h_resOPPQI
        //           << " "
        //           //<< "\nResult OPPScalar: " << resOPP1Scalar
        //           << "\nResult NPP: " << h_resNPPMean << "\nRef host: " /*<< test1 << " " << test2*/ << std::endl;

        const float minOPP = *std::min_element(runtimeOPP.begin(), runtimeOPP.end());
        const float maxOPP = *std::max_element(runtimeOPP.begin(), runtimeOPP.end());
        const float minNPP = *std::min_element(runtimeNPP.begin(), runtimeNPP.end());
        const float maxNPP = *std::max_element(runtimeNPP.begin(), runtimeNPP.end());

        const float sumOPP = std::accumulate(runtimeOPP.begin(), runtimeOPP.end(), 0.0f);
        const float sumNPP = std::accumulate(runtimeNPP.begin(), runtimeNPP.end(), 0.0f);

        const float sumSqrOPP = std::inner_product(runtimeOPP.begin(), runtimeOPP.end(), runtimeOPP.begin(), 0.0f);
        const float sumSqrNPP = std::inner_product(runtimeNPP.begin(), runtimeNPP.end(), runtimeNPP.begin(), 0.0f);

        float stdOPP  = 0;
        float stdNPP  = 0;
        float meanOPP = sumOPP;
        float meanNPP = sumNPP;

        if constexpr (iterations > 1)
        {
            stdOPP  = std::sqrt((sumSqrOPP - (sumOPP * sumOPP) / iterations) / (iterations - 1.0f));
            stdNPP  = std::sqrt((sumSqrNPP - (sumNPP * sumNPP) / iterations) / (iterations - 1.0f));
            meanOPP = sumOPP / iterations;
            meanNPP = sumNPP / iterations;
        }

        std::cout << "OPP: " << runtimeGlobalOPP << " ms Min: " << minOPP << " Max: " << maxOPP << " Mean: " << meanOPP
                  << " Std: " << stdOPP << std::endl;
        std::cout << "NPP: " << runtimeGlobalNPP << " ms Min: " << minNPP << " Max: " << maxNPP << " Mean: " << meanNPP
                  << " Std: " << stdNPP << std::endl;
        std::cout << "Done!" << std::endl;
    }
    catch (opp::OPPException &ex)
    {
        std::cout << ex.what();
    }
    catch (...)
    {
        return 1;
    }
    return 0;
}
