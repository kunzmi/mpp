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
#include <backends/npp/image/image16s.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image16sC1View.h> // NOLINT(misc-include-cleaner)
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

#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <numbers>

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

        constexpr size_t iterations = 100;
        constexpr size_t repeats    = 10; /**/
        constexpr int imgWidth      = 256 * 1;
        constexpr int imgHeight     = 256 * 1;
        // constexpr int sizeTpl       = 16;
        //  INT_MAX / 1024 / 256 + 1;

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        // cpu::Image<Pixel8uC1> cpu_src1(imgWidth, imgHeight);
        //                       cpu::ImageView<Pixel8u3> cpu_src1A(cpu_src1);
        /*cpu::Image<Pixel8uC1> cpu_src1 =
            cpu::Image<Pixel8uC1>::Load("C:\\Users\\kunz_\\source\\repos\\oppV1\\opp\\test\\testData\\bird256bw.tif");
        cpu::Image<Pixel8uC1> cpu_src2 = cpu::Image<Pixel8uC1>::Load(
            "C:\\Users\\kunz_\\source\\repos\\oppV1\\opp\\test\\testData\\bird256bwnoisy.tif");*/
        cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load("F:\\ogpp\\muster.tif");
        cpu::Image<Pixel8uC1> cpu_tpl  = cpu::Image<Pixel8uC1>::Load("F:\\ogpp\\template.tif");
        cpu::Image<Pixel8uC1> cpu_dst(imgWidth, imgHeight);
        cpu::Image<Pixel8uC1> cpu_temp(imgWidth, imgHeight);

        Image<Pixel8uC1> opp_src1(imgWidth, imgHeight);
        Image<Pixel8uC1> opp_tpl(16, 16);
        Image<Pixel8uC1> opp_temp(imgWidth, imgHeight);
        Image<Pixel8uC1> opp_dst(imgWidth, imgHeight);
        Image<Pixel32fC2> opp_sqr(imgWidth, imgHeight);
        DevVar<Pixel64fC1> meanTpl(1);
        DevVar<Pixel8uC1> opp_tpl2(16 * 16);

        nv::Image8uC1 npp_src1(imgWidth, imgHeight);
        nv::Image8uC1 npp_tpl(16, 16);
        nv::Image8uC1 npp_dst(imgWidth, imgHeight);
        DevVar<byte> npp_tpl2(16 * 16);

        /*cpu_src1.SetRoi(Roi(116, 65, sizeTpl, sizeTpl));
        cpu_src1.Copy(cpu_tpl);
        cpu_src1.ResetRoi();*/

        // constexpr int filterSize = 11;
        //  DevVar<float> filter(to_size_t(filterSize));
        //  DevVar<float> filter2(64 * 64);
        //  std::vector<float> filter_h(to_size_t(filterSize));
        //  std::vector<float> filter2_h(64 * 64, 1.0f / (9.0f));
        //  std::vector<Pixel8uC1> mask_h   = {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        //  0}; std::vector<byte> mask_bh       = {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
        //  0, 0}; std::vector<Pixel32sC1> maski_h = {0,  0,  10, 0,   0,   0,   10, 10, 10, 0,   10, 10, 10,
        //                                     10, 10, 0,  -10, -10, -10, 0,  0,  0,  -10, 0,  0};
        //  std::vector<int> maski_bh       = {0,  0,  -10, 0,  0,  0,  -10, -10, -10, 0,  10, 10, 10,
        //                                     10, 10, 0,   10, 10, 10, 0,   0,   0,   10, 0,  0};
        //  DevVar<Pixel8uC1> mask(to_size_t(filterSize * filterSize));
        //  DevVar<byte> maskb(to_size_t(filterSize * filterSize));
        //  DevVar<Pixel32sC1> maski(to_size_t(filterSize * filterSize));
        //  DevVar<int> maskib(to_size_t(filterSize * filterSize));
        //  DevVar<double> meanTpl(1);
        //  mask << mask_h;
        //  maskb << mask_bh;
        //  maski << maski_h;
        //  maskib << maski_bh;
        //  constexpr double sigma = 1.5; // 0.4 + (to_double(filterSize) / 3.0) * 0.6

        // float sum_filter = 0;
        // for (size_t i = 0; i < to_size_t(filterSize); i++)
        //{
        //     double x    = to_double(i) - to_double(filterSize / 2);
        //     filter_h[i] = to_float(1.0 / (std::sqrt(2.0 * std::numbers::pi_v<double>) * sigma) *
        //                            std::exp(-(x * x) / (2.0 * sigma * sigma)));
        //     sum_filter += filter_h[i];
        // }
        // for (size_t i = 0; i < to_size_t(filterSize); i++)
        //{
        //     filter_h[i] /= sum_filter;
        // }

        // filter << filter_h;
        // filter2 << filter2_h;

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

        NppStreamContext nppCtx;
        NppStatus status = nppGetStreamContext(&nppCtx);
        if (status != NPP_SUCCESS)
        {
            return -1;
        }

        StreamCtx oppCtx = StreamCtxSingleton::Get();

        cpu_src1 >> npp_src1;
        cpu_tpl >> npp_tpl;

        cpu_src1 >> opp_src1;
        cpu_tpl >> opp_tpl;

        opp_tpl2 << cpu_tpl.Pointer();
        npp_tpl2 << cpu_tpl.Pointer();
        //  cpu_mask >> opp_mask;
        //  cpu_mask >> npp_mask;

        DevVar<byte> buffer1(opp_tpl.MeanBufferSize(oppCtx));
        DevVar<byte> buffer2(npp_src1.MorphGetBufferSize());
        /*DevVar<byte> bufferMean(opp_tpl.MeanBufferSize(oppCtx));*/
        // DevVar<byte> bufferSSIM(opp_src1.SqrIntegralBufferSize(opp_dstSum, opp_dstSqr, oppCtx));

        // opp_src1.SetRoi(Border(-1, 0));
        // opp_dstAngle.Set(Pixel32fC1(0.0f), oppCtx);
        // npp_dstAngle.Set(Pixel32fC1(0.0f), nppCtx);
        /*opp_src1.SetRoi(-8);
        opp_dstAngle.SetRoi(-8);

        // npp_src1.SetRoi(-8);
        npp_dstAngle.SetRoi(-8); */
        // const BorderType opp_boder = BorderType::Replicate;
        /*const NppiMaskSize nppMask  = NppiMaskSize::NPP_MASK_SIZE_3_X_3;
        const MaskSize oppMask      = MaskSize::Mask_3x3;
        const FixedFilter oppFilter = FixedFilter::Laplace;*/
        // NppiSize nppMaskSize    = {filterSize, filterSize};
        // NppiPoint nppMaskCenter = {filterSize / 2, filterSize / 2};
        // Vec2i oppMaskSize   = {filterSize, filterSize};
        // Vec2i oppMaskCenter = {filterSize / 2, filterSize / 2};
        // DevVar<float> preComputedDist(64 * 64);
        // DevVar<float> preComputedDist2(64 * 64);
        // DevVar<Pixel32fC3> ssimo(1);
        // ssimo.Memset(0);
        // DevVar<float> ssim(1);
        // DevVar<float> mssim(3);
        // DevVar<float> wmssim(1);
        // DevVar<Pixel64fC1> qio(1);
        // DevVar<float> qin(1);

        // std::vector<float> preComputedH(64 * 64);
        // std::vector<float> preComputedH2(64 * 64);
        // const int maskSizeBilateral = 2;
        // const float posSquareSigma  = 15.0f;
        // for (int y = -maskSizeBilateral; y <= maskSizeBilateral; y++)
        //{
        //     const int idxY = y + maskSizeBilateral;
        //     for (int x = -maskSizeBilateral; x <= maskSizeBilateral; x++)
        //     {
        //         const int idxX = x + maskSizeBilateral;
        //         const int idx  = idxY * (maskSizeBilateral * 2 + 1) + idxX;

        //        const float fx = static_cast<float>(x);
        //        const float fy = static_cast<float>(y);
        //        float distSqr  = fx * fx + fy * fy;

        //        preComputedH[to_size_t(idx)] = std::exp(-distSqr / (2.0f * posSquareSigma));
        //        /*preComputedH[to_size_t(idx)] = 0;
        //        if (y == 0 && x == 0)
        //        {
        //            preComputedH[to_size_t(idx)] = 1;
        //        }*/
        //    }
        //}
        // preComputedDist << preComputedH;

        // float noise         = 100.0f;

        // npp_src1.FilterBorder32f(npp_dst, filter, {15, 15}, {6, 6}, NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);

        // opp_src1.ThresholdAdaptiveBoxFilter(opp_dst, oppMaskSize, oppMaskCenter, 0.5f, 255, 0, opp_boder, {0}, Roi(),
        //                                     oppCtx);
        // opp_src1.SeparableFilter(opp_dst, filter, oppMaskSize.x, oppMaskCenter.x, opp_boder, {0}, Roi(), oppCtx);
        // opp_src1.MaxFilter(opp_dst, oppMaskSize, opp_boder, {0}, Roi(), oppCtx);

        // opp_src1.GradientVectorSobel(opp_null16s, opp_null16s, opp_dstMag, opp_dstAngle, opp_null32fC4, Norm::L2,
        //                              MaskSize::Mask_5x5, opp_boder, {0}, Roi(), oppCtx);
        // opp_dstMag.CannyEdge(opp_dstAngle, opp_temp, opp_dst, 400, 3000, Roi(), oppCtx);
        // Pixel64fC1 ssimcpu;
        // cpu_src1.SSIM(cpu_src2, ssimcpu, 1);

        // opp_tpl2.Div(Pixel32fC1(255), oppCtx);
        // opp_src2.Div(Pixel32fC1(255), oppCtx);

        opp_src1.BlackHat(opp_temp, opp_dst, opp_tpl2, 16, BorderType::Replicate, oppCtx);
        npp_src1.MorphBlackHatBorder(npp_dst, npp_tpl2, {16, 16}, {8, 8}, buffer2, NPP_BORDER_REPLICATE, nppCtx);
        cpu_src1.BlackHat(cpu_temp, cpu_dst, cpu_tpl.Pointer(), 16, BorderType::Replicate);

        // npp_src1.MSSSIM(npp_src2, mssim, bufferSSIM, nppCtx);
        // npp_src1.WMSSSIM(npp_src2, wmssim, bufferSSIM, nppCtx);
        // npp_src1.QualityIndex(npp_src2, {imgWidth, imgHeight}, qin, bufferSSIM, nppCtx);

        /*DevVarView<Pixel64fC1> tempvar(reinterpret_cast<Pixel64fC1 *>(meanTpl.Pointer()), 8);
        opp_tpl.Mean(tempvar, bufferMean, oppCtx);
        opp_src1.BoxAndSumSquareFilter(opp_boxFiltered, sizeTpl, BorderType::Constant, {0}, Roi(), oppCtx);
        opp_src1.CrossCorrelationCoefficient(opp_boxFiltered, opp_tpl, meanTpl, opp_dstAngle, BorderType::Constant, {0},
                                             Roi(), oppCtx);*/

        //  opp_src1.FixedFilter(opp_dst, oppFilter, oppMask, opp_boder, {0}, Roi(), oppCtx);
        //  npp_src1.FilterLaplaceBorder(npp_dst, nppMask, NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);
        /*npp_src1.FilterColumnBorder32f(npp_dst, filter, filterSize, filterSize / 2,
                                       NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);
        npp_src1.FilterBoxBorderAdvanced(npp_dst, nppMaskSize, nppMaskCenter, NppiBorderType::NPP_BORDER_REPLICATE,
                                         buffer, nppCtx);*/
        /*npp_src1.FilterMaxBorder(npp_dst, nppMaskSize, nppMaskCenter, NppiBorderType::NPP_BORDER_REPLICATE,
                                 nppCtx); */
        // npp_src1.Dilate3x3Border(npp_dst, NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);

        // npp_src1.CrossCorrSame_NormLevelAdvanced(npp_tpl, npp_dstAngle, buffer1, buffer2, nppCtx);

        /*cpu_dst << opp_temp;
        cpu_dst.Save("f:\\cannyTempOpp.tif");
        cpu_dstMag << opp_dstMag;
        cpu_dstMag.Save("f:\\gradientVectorMagOpp.tif");
        cpu_dstAngle << opp_dstAngle;
        cpu_dstAngle.Save("f:\\gradientVectorAngleOpp.tif");

        nv::ImageView<Pixel8uC1> npp_temp0(reinterpret_cast<Pixel8uC1 *>(buffer.Pointer()),
                                           SizePitched({256, 256}, 256), npp_dst.ROI());
        nv::ImageView<Pixel16sC1> npp_temp1(reinterpret_cast<Pixel16sC1 *>(buffer.Pointer() + 256 * 256),
                                            SizePitched({256, 256}, 256 * 2), npp_dst.ROI());
        nv::ImageView<Pixel32fC1> npp_temp2(reinterpret_cast<Pixel32fC1 *>(buffer.Pointer() + 256 * 256 * 3),
                                            SizePitched({256, 256}, 256 * 4), npp_dst.ROI());
        nv::ImageView<Pixel16sC1> npp_temp3(reinterpret_cast<Pixel16sC1 *>(buffer.Pointer() + 256 * 256 * 7),
                                            SizePitched({256, 256}, 256 * 2), npp_dst.ROI());*/

        /*cpu_dst << npp_temp0;
        cpu_dst.Save("f:\\cannyTemp0Npp.tif");
        cpu_dstX << npp_temp1;
        cpu_dstX.Save("f:\\cannyTemp1Npp.tif");
        cpu_dstY << npp_temp3;
        cpu_dstY.Save("f:\\cannyTemp3Npp.tif");
        cpu_dstAngle << npp_dstAngle;
        cpu_dstAngle.Save("f:\\crossCorrNpp.tif");
        cpu_dstAngle << opp_dstAngle;
        cpu_dstAngle.Save("f:\\crossCorrOpp.tif");*/

        // cpu_src1.Save("f:\\lowpassOrig.tif");
        cpu_dst.Save("f:\\filterbhCpu.tif");
        cpu_dst << opp_dst;
        cpu_dst.Save("f:\\filterbhOpp.tif");
        cpu_dst << npp_dst;
        cpu_dst.Save("f:\\filterbhNpp.tif"); /**/

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

        std::cout << "Now running OPP:" << std::endl;
        startGlobalOPP.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIterOPP.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                /*opp_tpl.Mean(tempvar, bufferMean, oppCtx);
                opp_src1.BoxAndSumSquareFilter(opp_boxFiltered, sizeTpl, BorderType::Constant, {0}, Roi(), oppCtx);
                opp_src1.CrossCorrelationCoefficient(opp_boxFiltered, opp_tpl, meanTpl, opp_dstAngle,
                                                     BorderType::Constant, {0}, Roi(), oppCtx); */
                // opp_src1.SqrIntegral(opp_dstSum, opp_dstSqr, 0, 0, bufferSSIM, oppCtx);
                // opp_dstStd.SetRoi(Roi(0, 0, imgWidth - filterSize, imgHeight - filterSize));
                //  opp_dstSum.RectStdDev(opp_dstSqr, opp_dstStd, FilterArea(filterSize, 0), oppCtx);
                // opp_dstSum.ResetRoi();
                // opp_dstSqr.ResetRoi();
                // opp_dstStd.ResetRoi();
            }
            endIterOPP.Record();
            endIterOPP.Synchronize();
            const float runtime = endIterOPP - startIterOPP;
            runtimeOPP[i]       = runtime;
        }
        endGlobalOPP.Record();
        endGlobalOPP.Synchronize();
        const float runtimeGlobalOPP = endGlobalOPP - startGlobalOPP;

        std::cout << "OPP done, now running NPP:" << std::endl;

        startGlobalNPP.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIterNPP.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                // npp_src1.FilterGaussAdvancedBorder(npp_dst, filterSize, filter, NppiBorderType::NPP_BORDER_REPLICATE,
                //                                    nppCtx);
                // npp_src1.FilterBoxBorder(npp_dst, nppMaskSize, nppMaskCenter, NppiBorderType::NPP_BORDER_REPLICATE,
                //                         nppCtx);
                // npp_src1.FilterRowBorder32f(npp_dst, filter, filterSize, filterSize / 2,
                //                            NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);
                /*npp_src1.FilterMinBorder(npp_dst, nppMaskSize, nppMaskCenter, NppiBorderType::NPP_BORDER_REPLICATE,
                                         nppCtx);*/
                /*npp_src1.FilterMaxBorder(npp_dst, nppMaskSize, nppMaskCenter, NppiBorderType::NPP_BORDER_REPLICATE,
                                         nppCtx); */
                // npp_src1.SqrIntegral(npp_dstSum, npp_dstSqr, 0, 0, nppCtx);
                // npp_dstSum.SetRoi(Roi(0, 0, imgWidth - filterSize, imgHeight - filterSize));
                // npp_dstSqr.SetRoi(Roi(0, 0, imgWidth - filterSize, imgHeight - filterSize));
                // npp_dstStd.SetRoi(Roi(0, 0, imgWidth - filterSize, imgHeight - filterSize));
                //  npp_dstSum.RectStdDev(npp_dstSqr, npp_dstStd, NppiRect{0, 0, filterSize, filterSize}, nppCtx);
                // npp_dstSum.ResetRoi();
                // npp_dstSqr.ResetRoi();
                // npp_dstStd.ResetRoi();
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
