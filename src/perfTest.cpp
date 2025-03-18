#include "backends/npp/image/imageView.h"
#include "common/exception.h"
#include "common/image/pixelTypes.h"
#include <algorithm>
#include <backends/cuda/devVar.h>     // NOLINT(misc-include-cleaner)
#include <backends/cuda/devVarView.h> // NOLINT(misc-include-cleaner)
#include <backends/cuda/event.h>
#include <backends/cuda/image/image.h>     // NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView.h> // NOLINT(misc-include-cleaner)
#include <backends/cuda/image/statistics/sum.h>
#include <backends/cuda/stream.h>
#include <backends/cuda/streamCtx.h>
#include <backends/npp/image/image16u.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image16uC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32f.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32fC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image8u.h>        // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image8uC1View.h>  // NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <cmath>
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
        constexpr size_t iterations = 1;
        constexpr size_t repeats    = 1;
        constexpr int imgSize       = 2048 * 1;

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        // const std::filesystem::path baseDir = std::filesystem::path(PROJECT_SOURCE_DIR) / "test/testData";

        // auto cpu_flower = cpu::Image<Pixel8uC3>::Load(baseDir / "flower.tif");

        /*cpu_res.Save(baseDir / "resCPU.tif");
        resGPU.Save(baseDir / "resGPU.tif");

        cpu_res.ResetRoi();
        resGPU.ResetRoi();
        bool issame = cpu_res.IsIdentical(resGPU);

        std::cout << "Images are identical: " << std::boolalpha << issame << std::endl;*/

        cpu::Image<Pixel8uC1> cpu_src1(imgSize, imgSize);
        // cpu::Image<Pixel8uC1> cpu_src2(imgSize, imgSize);
        // cpu::Image<Pixel32fC1> cpu_dst(1, imgSize);
        // cpu::Image<Pixel32fC1> cpu_dst_opp(1, imgSize);
        // cpu::Image<Pixel32fC1> cpu_dst_npp(1, imgSize);

        Image<Pixel8uC1> opp_src1(imgSize, imgSize);
        // Image<Pixel8uC1> opp_src2(imgSize, imgSize);
        // Image<Pixel32fC1> opp_dst(1, imgSize);

        nv::Image8uC1 npp_src1(imgSize, imgSize);
        // nv::Image8uC1 npp_src2(imgSize, imgSize);
        // nv::Image64fC1 npp_dst(1, imgSize);

        cpu_src1.FillRandom();
        cpu_src1.Set(1);
        // cpu_dst.FillRandom();

        /*cpu_src1.Set({0});
        cpu_src2.Set({0});*/

        cpu_src1 >> opp_src1;
        // cpu_src2 >> opp_src2;

        /*cpu::ImageView<Pixel8uC4> cpu_src1NoA = cpu_src1;
        cpu::ImageView<Pixel8uC4> cpu_src2NoA = cpu_src2;
        cpu_src1NoA >> npp_src1;
        cpu_src2NoA >> npp_src2;*/
        cpu_src1 >> npp_src1;
        // cpu_src2 >> npp_src2;

        NppStreamContext nppCtx;
        NppStatus status = nppGetStreamContext(&nppCtx);
        if (status != NPP_SUCCESS)
        {
            return -1;
        }

        StreamCtx oppCtx = StreamCtxSingleton::Get();

        DevVar<Pixel32fC1> bufferOpp(imgSize);
        DevVar<Pixel64fC1> sumOpp(1);
        DevVar<double> sumNpp(1);
        DevVar<byte> bufferNpp(npp_src1.SumGetBufferHostSize(nppCtx));

        // warm up
        npp_src1.Sum(bufferNpp, sumNpp, nppCtx);

        opp::image::cuda::InvokeSumSrc(opp_src1.PointerRoi(), opp_src1.Pitch(), bufferOpp.Pointer(), sumOpp.Pointer(),
                                       opp_src1.SizeRoi(), oppCtx); /**/

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

        /*opp_src1.SetRoi(Border(-1, 0));
        npp_src1.SetRoi(Border(-1, 0));
        opp_src2.SetRoi(-2);
        opp_dst.SetRoi(-2);

        npp_src1.SetRoi(-2);
        npp_src2.SetRoi(-2);
        npp_dst.SetRoi(-2); */

        startGlobalOPP.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIterOPP.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                opp::image::cuda::InvokeSumSrc(opp_src1.PointerRoi(), opp_src1.Pitch(), bufferOpp.Pointer(),
                                               sumOpp.Pointer(), opp_src1.SizeRoi(), oppCtx);
                // opp_src1.AddSquare(opp_dst, oppCtx);
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
                npp_src1.Sum(bufferNpp, sumNpp, nppCtx);
            }
            endIterNPP.Record();
            endIterNPP.Synchronize();
            const float runtime = endIterNPP - startIterNPP;
            runtimeNPP[i]       = runtime;
        }
        endGlobalNPP.Record();
        endGlobalNPP.Synchronize();
        const float runtimeGlobalNPP = endGlobalNPP - startGlobalNPP;

        double resNPP{};
        Pixel64fC1 resOPP{};
        double test = 0;
        sumNpp >> resNPP;
        sumOpp >> resOPP;

        std::vector<Pixel32fC1> sums(imgSize);
        bufferOpp.CopyToHost(sums.data());

        for (const auto &elem : sums)
        {
            test += elem.x;
        }

        std::cout << "Result OPP: " << to_size_t(resOPP.x) << " Result NPP: " << to_size_t(resNPP)
                  << " Ref host: " << to_size_t(test) << std::endl;

        // cpu::ImageView<Pixel8uC4> cpu_dst_nppNoA = cpu_dst_npp;
        /*cpu_dst_npp << npp_dst;
        cpu_dst_opp << opp_dst;*/

        // const bool ok = cpu_dst_opp.IsIdentical(cpu_dst_npp);

        // std::cout << std::boolalpha << "Same result: " << ok << std::endl;

        const float minOPP = *std::min_element(runtimeOPP.begin(), runtimeOPP.end());
        const float maxOPP = *std::max_element(runtimeOPP.begin(), runtimeOPP.end());
        const float minNPP = *std::min_element(runtimeNPP.begin(), runtimeNPP.end());
        const float maxNPP = *std::max_element(runtimeNPP.begin(), runtimeNPP.end());

        const float sumOPP = std::accumulate(runtimeOPP.begin(), runtimeOPP.end(), 0.0f);
        const float sumNPP = std::accumulate(runtimeNPP.begin(), runtimeNPP.end(), 0.0f);

        const float sumSqrOPP = std::inner_product(runtimeOPP.begin(), runtimeOPP.end(), runtimeOPP.begin(), 0.0f);
        const float sumSqrNPP = std::inner_product(runtimeNPP.begin(), runtimeNPP.end(), runtimeNPP.begin(), 0.0f);

        const float stdOPP = sumSqrOPP;
        // std::sqrt((sumSqrOPP - (sumOPP * sumOPP) / iterations) / (iterations - 1.0f));
        const float stdNPP = sumSqrNPP;
        // std::sqrt((sumSqrNPP - (sumNPP * sumNPP) / iterations) / (iterations - 1.0f));
        const float meanOPP = sumOPP;
        // sumOPP / iterations;
        const float meanNPP = sumNPP;
        // sumNPP / iterations;

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
