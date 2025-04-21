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
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fC1View.h> // NOLINT(misc-include-cleaner)
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
        /*constexpr size_t iterations = 1;
        constexpr size_t repeats    = 1;*/
        constexpr int imgWidth  = 256 * 1;
        constexpr int imgHeight = 256 * 1;
        // INT_MAX / 1024 / 256 + 1;

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        // const std::filesystem::path baseDir = std::filesystem::path(PROJECT_SOURCE_DIR) / "test/testData";

        // auto cpu_flower = cpu::Image<Pixel8uC3>::Load(baseDir / "flower.tif");

        /*cpu_res.Save(baseDir / "resCPU.tif");
        resGPU.Save(baseDir / "resGPU.tif");

        cpu_res.ResetRoi();
        resGPU.ResetRoi();
        bool issame = cpu_res.IsIdentical(resGPU);

        std::cout << "Images are identical: " << std::boolalpha << issame << std::endl;*/

        cpu::Image<Pixel8uC1> cpu_src1(imgWidth, imgHeight);
        /*cpu::Image<Pixel8uC1> cpu_src2(imgWidth, imgHeight);
        cpu::Image<Pixel32uC1> cpu_src22(imgHeight + 1, imgWidth + 1);
        cpu::Image<Pixel64uC1> cpu_src3(imgWidth + 1, imgHeight + 1);
        cpu::Image<Pixel64uC1> cpu_src32(imgHeight + 1, imgWidth + 1);
        cpu::Image<Pixel8uC1> cpu_mask(imgWidth, imgHeight);*/

        /*cpu::Image<Pixel32sC1> cpu_src4(imgWidth + 1, imgHeight + 1);
        cpu::Image<Pixel64fC1> cpu_src5(imgHeight + 1, imgWidth + 1);*/
        // cpu::Image<Pixel32fC1> cpu_dst(1, imgHeight);
        // cpu::Image<Pixel32fC1> cpu_dst_opp(1, imgHeight);
        // cpu::Image<Pixel32fC1> cpu_dst_npp(1, imgHeight);

        Image<Pixel8uC1> opp_src1(imgWidth, imgHeight);
        /*Image<Pixel8uC1> opp_src2(imgWidth, imgHeight);
        Image<Pixel32uC1> opp_src22(imgHeight + 1, imgWidth + 1);
        Image<Pixel64uC1> opp_src3(imgWidth + 1, imgHeight + 1);
        Image<Pixel64uC1> opp_src32(imgHeight + 1, imgWidth + 1);*/
        // Image<Pixel32fC1> opp_dst(1, imgHeight);

        nv::Image8uC1 npp_src1(imgWidth, imgHeight);
        /* nv::Image8uC1 npp_src2(imgWidth, imgHeight);
         nv::Image64fC1 npp_src3(imgWidth + 1, imgHeight + 1);*/
        //   nv::Image64fC1 npp_dst(1, imgHeight);

        for (auto &pixel : cpu_src1)
        {
            pixel.Value().x = to_byte(pixel.Pixel().x % 256);
        } /**/

        // cpu_src1.Set(1);
        // cpu_src1.FillRandom(0);
        //  cpu_src2.FillRandom(1);
        //  cpu_src2.Div(4).Add(cpu_src1).Sub(2); /**/
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

        const size_t bufferSizeOpp  = opp_src1.HistogramEvenBufferSize(21, oppCtx);
        const size_t bufferSizeOpp2 = opp_src1.HistogramRangeBufferSize(21, oppCtx);
        const size_t bufferSizeNpp  = npp_src1.HistogramEvenGetBufferSize(21, nppCtx);

        DevVar<byte> bufferOpp1(bufferSizeOpp);
        DevVar<byte> bufferOpp2(bufferSizeOpp2);
        DevVar<byte> bufferNpp1(bufferSizeNpp);
        DevVar<int> histNpp(20);
        DevVar<int> histOpp(20);
        DevVar<int> histOpp1(20);
        DevVar<int> histOpp2(20);

        DevVar<int> dLevelsNpp(21);
        DevVar<int> dLevelsOpp(21);
        std::vector<int> hLevelsNpp(21);
        std::vector<int> hLevelsOpp(21);
        std::vector<int> hLevelsNpp2 = npp_src1.EvenLevels(21, 0, 256);

        opp_src1.EvenLevels(hLevelsNpp.data(), 21, 0, 256, opp::HistorgamEvenMode::NPP);
        opp_src1.EvenLevels(hLevelsOpp.data(), 21, 0, 256);

        dLevelsNpp << hLevelsNpp;
        dLevelsOpp << hLevelsOpp;

        opp_src1.HistogramEven(histOpp, 0, 256, bufferOpp1, oppCtx);
        npp_src1.HistogramEven(histNpp, 20, 0, 256, bufferNpp1, nppCtx);
        opp_src1.HistogramRange(histOpp1, dLevelsNpp, bufferOpp2, oppCtx);
        opp_src1.HistogramRange(histOpp2, dLevelsOpp, bufferOpp2, oppCtx);

        std::vector<int> h_histOpp(20);
        std::vector<int> h_histOpp1(20);
        std::vector<int> h_histOpp2(20);
        std::vector<int> h_histNpp(20);

        histOpp >> h_histOpp;
        histOpp1 >> h_histOpp1;
        histOpp2 >> h_histOpp2;
        histNpp >> h_histNpp;

        std::cout << "OPP\tNPP\tOPP1\tOPP2\n";
        for (size_t i = 0; i < 20; i++)
        {
            std::cout << h_histOpp[i] << "\t" << h_histNpp[i] << "\t" << h_histOpp1[i] << "\t" << h_histOpp2[i]
                      << std::endl;
        }

        /*DevVar<Pixel64uC1> bufferOpp2(imgHeight);
        DevVar<Pixel64uC1> bufferOpp3(imgHeight);
        DevVar<Pixel64uC1> bufferOpp4(imgHeight);
        DevVar<Pixel64uC1> bufferOpp5(imgHeight); */
        // DevVar<ulong64> bufferOppMask(imgHeight);
        // DevVar<Pixel64fC1> resOppQI(1);
        // DevVar<Pixel64fC1> resOppStd(1);
        // DevVar<double> resOppScalarMean(1);
        // DevVar<double> resOppScalarStd(1);

        // DevVar<float> resNpp1(1);
        //   DevVar<double> resNpp2(1);
        // DevVar<byte> bufferNpp(npp_src1.SSIMGetBufferHostSize(nppCtx));
        //   opp_src1.SetRoi(Border{-1, -1, 0, 0});
        //    warm up
        // npp_src1.SSIM(npp_src2, resNpp1, bufferNpp, nppCtx);

        /*opp::image::cuda::InvokeSSIMSrcSrc(opp_src1.PointerRoi(), opp_src1.Pitch(), opp_src2.PointerRoi(),
                                           opp_src2.Pitch(), bufferOpp1.Pointer(), bufferOpp2.Pointer(),
                                           bufferOpp3.Pointer(), bufferOpp4.Pointer(), bufferOpp5.Pointer(),
                                           resOppQI.Pointer(), 1, 0.01, 0.03, opp_src2.SizeRoi(), oppCtx);*/
        /*opp::image::cuda::InvokeIntegralSqrSrc(opp_src1.PointerRoi(), opp_src1.Pitch(), opp_src22.PointerRoi(),
                                               opp_src22.Pitch(), opp_src32.PointerRoi(), opp_src32.Pitch(),
                                               opp_src2.PointerRoi(), opp_src2.Pitch(), opp_src3.PointerRoi(),
                                               opp_src3.Pitch(), {0}, {0}, opp_src2.SizeRoi(), oppCtx);*/

        /*float hnpp;
        resNpp1 >> hnpp;

        Pixel64fC1 hopp;
        resOppQI >> hopp;

        std::cout << hopp.x << " " << hnpp;*/

        /*cpu_src2 << opp_src2;

        cpu_src3 << opp_src3;
        cpu_src4 << npp_src2;

        cpu_src5 << npp_src3;*/

        /*cpu_src2.Save("f:\\integral.tif");
        cpu_src3.Save("f:\\integralSqr.tif");
        cpu_src4.Save("f:\\integralNpp.tif");
        cpu_src5.Save("f:\\integralNppSqr.tif");*/

        /////*std::cout << "start sleeping" << std::endl;
        //// std::this_thread::sleep_for(std::chrono::seconds(5));
        //// std::cout << "...DONE" << std::endl;*/
        // std::vector<float> runtimeOPP(iterations);
        // std::vector<float> runtimeNPP(iterations);

        // const Event startGlobalOPP;
        // const Event startIterOPP;
        // const Event endIterOPP;
        // const Event endGlobalOPP;
        // const Event startGlobalNPP;
        // const Event startIterNPP;
        // const Event endIterNPP;
        // const Event endGlobalNPP;

        ///*opp_src1.SetRoi(Border(-1, 0));
        // npp_src1.SetRoi(Border(-1, 0));
        // opp_src2.SetRoi(-2);
        // opp_dst.SetRoi(-2);

        // npp_src1.SetRoi(-2);
        // npp_src2.SetRoi(-2);
        // npp_dst.SetRoi(-2); */

        // startGlobalOPP.Record();
        // for (size_t i = 0; i < iterations; i++)
        //{
        //     startIterOPP.Record();
        //     for (size_t r = 0; r < repeats; r++)
        //     {
        //         /*opp::image::cuda::InvokeSSIMSrcSrc(opp_src1.PointerRoi(), opp_src1.Pitch(), opp_src2.PointerRoi(),
        //                                            opp_src2.Pitch(), bufferOpp1.Pointer(), bufferOpp2.Pointer(),
        //                                            bufferOpp3.Pointer(), bufferOpp4.Pointer(), bufferOpp5.Pointer(),
        //                                            resOppQI.Pointer(), 1, 0.01, 0.03, opp_src2.SizeRoi(), oppCtx);*/
        //     }
        //     endIterOPP.Record();
        //     endIterOPP.Synchronize();
        //     const float runtime = endIterOPP - startIterOPP;
        //     runtimeOPP[i]       = runtime;
        // }
        // endGlobalOPP.Record();
        // endGlobalOPP.Synchronize();
        // const float runtimeGlobalOPP = endGlobalOPP - startGlobalOPP;

        // startGlobalNPP.Record();
        // for (size_t i = 0; i < iterations; i++)
        //{
        //     startIterNPP.Record();
        //     for (size_t r = 0; r < repeats; r++)
        //     {
        //         //npp_src1.QualityIndex(npp_src2, npp_src2.NppiSizeRoi(), resNpp1, bufferNpp, nppCtx);
        //     }
        //     endIterNPP.Record();
        //     endIterNPP.Synchronize();
        //     const float runtime = endIterNPP - startIterNPP;
        //     runtimeNPP[i]       = runtime;
        // }
        // endGlobalNPP.Record();
        // endGlobalNPP.Synchronize();
        // const float runtimeGlobalNPP = endGlobalNPP - startGlobalNPP;

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

        // const float minOPP = *std::min_element(runtimeOPP.begin(), runtimeOPP.end());
        // const float maxOPP = *std::max_element(runtimeOPP.begin(), runtimeOPP.end());
        // const float minNPP = *std::min_element(runtimeNPP.begin(), runtimeNPP.end());
        // const float maxNPP = *std::max_element(runtimeNPP.begin(), runtimeNPP.end());

        // const float sumOPP = std::accumulate(runtimeOPP.begin(), runtimeOPP.end(), 0.0f);
        // const float sumNPP = std::accumulate(runtimeNPP.begin(), runtimeNPP.end(), 0.0f);

        // const float sumSqrOPP = std::inner_product(runtimeOPP.begin(), runtimeOPP.end(), runtimeOPP.begin(), 0.0f);
        // const float sumSqrNPP = std::inner_product(runtimeNPP.begin(), runtimeNPP.end(), runtimeNPP.begin(), 0.0f);

        // float stdOPP  = 0;
        // float stdNPP  = 0;
        // float meanOPP = sumOPP;
        // float meanNPP = sumNPP;

        // if constexpr (iterations > 1)
        //{
        //     stdOPP  = std::sqrt((sumSqrOPP - (sumOPP * sumOPP) / iterations) / (iterations - 1.0f));
        //     stdNPP  = std::sqrt((sumSqrNPP - (sumNPP * sumNPP) / iterations) / (iterations - 1.0f));
        //     meanOPP = sumOPP / iterations;
        //     meanNPP = sumNPP / iterations;
        // }

        // std::cout << "OPP: " << runtimeGlobalOPP << " ms Min: " << minOPP << " Max: " << maxOPP << " Mean: " <<
        // meanOPP
        //           << " Std: " << stdOPP << std::endl;
        // std::cout << "NPP: " << runtimeGlobalNPP << " ms Min: " << minNPP << " Max: " << maxNPP << " Mean: " <<
        // meanNPP
        //           << " Std: " << stdNPP << std::endl;
        // std::cout << "Done!" << std::endl;
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
