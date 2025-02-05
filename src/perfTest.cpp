#include "common/exception.h"
#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/event.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/stream.h>
#include <backends/cuda/streamCtx.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <cmath>
#include <common/version.h>
#include <cstddef>
#include <filesystem>
#include <ios>
#include <iostream>
#include <nppcore.h>
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

        constexpr size_t iterations = 100;
        constexpr size_t repeats    = 10;
        constexpr int imgSize       = 4096 * 2;

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        const std::filesystem::path baseDir = std::filesystem::path(PROJECT_SOURCE_DIR) / "test/testData";

        // auto cpu_flower = cpu::Image<Pixel8uC3>::Load(baseDir / "flower.tif");

        /*cpu_res.Save(baseDir / "resCPU.tif");
        resGPU.Save(baseDir / "resGPU.tif");

        cpu_res.ResetRoi();
        resGPU.ResetRoi();
        bool issame = cpu_res.IsIdentical(resGPU);

        std::cout << "Images are identical: " << std::boolalpha << issame << std::endl;*/

        cpu::Image<Pixel8uC4A> cpu_src1(imgSize, imgSize);
        cpu::Image<Pixel8uC4A> cpu_src2(imgSize, imgSize);
        cpu::Image<Pixel8uC4A> cpu_dst(imgSize, imgSize);
        cpu::Image<Pixel8uC4A> cpu_dst_opp(imgSize, imgSize);
        cpu::Image<Pixel8uC4A> cpu_dst_npp(imgSize, imgSize);

        Image<Pixel8uC4A> opp_src1(imgSize, imgSize);
        Image<Pixel8uC4A> opp_src2(imgSize, imgSize);
        Image<Pixel8uC4A> opp_dst(imgSize, imgSize);

        nv::Image8uC4 npp_src1(imgSize, imgSize);
        nv::Image8uC4 npp_src2(imgSize, imgSize);
        nv::Image8uC4 npp_dst(imgSize, imgSize);

        cpu_src1.FillRandom();
        cpu_src2.FillRandom();

        cpu_src1 >> opp_src1;
        cpu_src2 >> opp_src2;

        npp_src1 << cpu_src1.Pointer();
        npp_src2 << cpu_src2.Pointer();

        NppStreamContext nppCtx;
        NppStatus status = nppGetStreamContext(&nppCtx);
        if (status != NPP_SUCCESS)
        {
            return -1;
        }

        StreamCtx oppCtx = StreamCtxSingleton::Get();
        // warm up
        npp_src1.MulA(npp_src2, npp_dst, 2, nppCtx);
        opp_src1.Mul(opp_src2, opp_dst, 2, oppCtx);

        std::vector<float> runtimeOPP(iterations);
        std::vector<float> runtimeNPP(iterations);

        Event startGlobalOPP;
        Event startIterOPP;
        Event endIterOPP;
        Event endGlobalOPP;
        Event startGlobalNPP;
        Event startIterNPP;
        Event endIterNPP;
        Event endGlobalNPP;

        opp_src1.SetRoi(-2);
        opp_src2.SetRoi(-2);
        opp_dst.SetRoi(-2);

        npp_src1.SetRoi(-2);
        npp_src2.SetRoi(-2);
        npp_dst.SetRoi(-2); /**/

        startGlobalOPP.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIterOPP.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                opp_src1.Mul(opp_src2, opp_dst, 2, oppCtx);
            }
            endIterOPP.Record();
            endIterOPP.Synchronize();
            const float runtime = startIterOPP - endIterOPP;
            runtimeOPP[i]       = runtime;
        }
        endGlobalOPP.Record();
        endGlobalOPP.Synchronize();
        const float runtimeGlobalOPP = startGlobalOPP - endGlobalOPP;

        startGlobalNPP.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIterNPP.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                npp_src1.MulA(npp_src2, npp_dst, 2, nppCtx);
            }
            endIterNPP.Record();
            endIterNPP.Synchronize();
            const float runtime = startIterNPP - endIterNPP;
            runtimeNPP[i]       = runtime;
        }
        endGlobalNPP.Record();
        endGlobalNPP.Synchronize();
        const float runtimeGlobalNPP = startGlobalNPP - endGlobalNPP;

        npp_dst.CopyToHost(cpu_dst_npp.Pointer());
        cpu_dst_opp << opp_dst;

        const bool ok = cpu_dst_opp.IsIdentical(cpu_dst_npp);

        std::cout << std::boolalpha << "Same result: " << ok << std::endl;

        const float minOPP = *std::min_element(runtimeOPP.begin(), runtimeOPP.end());
        const float maxOPP = *std::max_element(runtimeOPP.begin(), runtimeOPP.end());
        const float minNPP = *std::min_element(runtimeNPP.begin(), runtimeNPP.end());
        const float maxNPP = *std::max_element(runtimeNPP.begin(), runtimeNPP.end());

        const float sumOPP = std::accumulate(runtimeOPP.begin(), runtimeOPP.end(), 0.0f);
        const float sumNPP = std::accumulate(runtimeNPP.begin(), runtimeNPP.end(), 0.0f);

        const float sumSqrOPP = std::inner_product(runtimeOPP.begin(), runtimeOPP.end(), runtimeOPP.begin(), 0.0f);
        const float sumSqrNPP = std::inner_product(runtimeNPP.begin(), runtimeNPP.end(), runtimeNPP.begin(), 0.0f);

        const float stdOPP  = std::sqrt((sumSqrOPP - (sumOPP * sumOPP) / iterations) / (iterations - 1.0f));
        const float stdNPP  = std::sqrt((sumSqrNPP - (sumNPP * sumNPP) / iterations) / (iterations - 1.0f));
        const float meanOPP = sumOPP / iterations;
        const float meanNPP = sumNPP / iterations;

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
