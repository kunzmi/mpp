
// NOLINTBEGIN

// #include <common/image/size2D.h>
#include <common/safeCast.h>
#include <common/version.h>
#include <cstddef>
// #include <cuda_runtime_api.h>
// #include <driver_types.h>
#include <backends/cuda/image/arithmetic/addSquareProductWeightedOutputType.h>
#include <backends/cuda/image/statistics/sum.h>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/functors/srcPlanar2Functor.h>
#include <common/image/functors/srcPlanar3Functor.h>
#include <common/image/functors/srcPlanar4Functor.h>
#include <common/image/functors/srcSingleChannelFunctor.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <iostream>
#include <vector>
// #include <common/arithmetic/ternary_operators.h>
// #include <common/arithmetic/unary_operators.h>
// #include <common/image/pixelTypes.h>
// #include <common/vector_typetraits.h>
// #include <common/vector4A.h>
// #include <common/vectorTypes.h>

// #include<backends / cuda / image / arithmetic / add.cu>
// #include <backends/cuda/image/arithmetic/add.h>
#include <backends/cuda/image/configurations.h>
// #include <backends/cuda/image/forEachPixelKernel.h>
//
#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/image/imageView_dataExchangeAndInit_impl.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/complex_impl.h>
#include <common/half_fp16.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/vector1.h>
#include <common/vector1_impl.h>
#include <common/vector2.h>
#include <common/vector2_impl.h>
#include <common/vector3.h>
#include <common/vector3_impl.h>
#include <common/vector4A_impl.h>
#include <common/vector4_impl.h>
#include <common/vectorTypes.h>
// #include <common/scratchBuffer.h>
// #include <half/half.hpp>

#include <common/arithmetic/binary_operators.h>

#include "common/defines.h"
#include "common/exception.h"
#include "common/utilities.h"
#include "common/vector4.h"
#include <common/image/pixelTypeEnabler.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <ios>
using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;
namespace cpu = opp::image::cpuSimple;

int main()
{
    // StreamCtx oppCtx = StreamCtxSingleton::Get();

    // c_double::Ln(cf);
    // constexpr bool iscf = NonNativeFloatingPoint<BFloat16>;
    /*
    constexpr bool val1 = std::is_trivially_assignable_v<Pixel32fC4, Pixel32fC4>;
    constexpr bool val2 = std::is_trivially_constructible_v<Pixel32fC2>;
    constexpr bool val3 = std::is_trivially_copyable_v<Pixel32fC4>;
    constexpr bool val4 = std::is_trivially_copy_assignable_v<Pixel32fC4>;
    constexpr bool val5 = std::is_trivially_copy_constructible_v<Pixel32fC4>;
    constexpr bool val6 = std::is_trivially_default_constructible_v<Pixel32fC4>;
    constexpr bool val6 = std::is_trivially_move_assignable_v<Pixel32fC4>;
    constexpr bool val6 = std::is_trivially_move_constructible_v<Pixel32fC4>;

    constexpr bool val1 = std::is_trivially_assignable_v<HalfFp16, HalfFp16>;
    constexpr bool val2 = std::is_trivially_constructible_v<HalfFp16>;
    constexpr bool val3 = std::is_trivially_copyable_v<HalfFp16>;
    constexpr bool val4 = std::is_trivially_copy_assignable_v<HalfFp16>;
    constexpr bool val5 = std::is_trivially_copy_constructible_v<HalfFp16>;
    constexpr bool val6 = std::is_trivially_default_constructible_v<HalfFp16>;
    constexpr bool val6 = std::is_trivially_move_assignable_v<HalfFp16>;
    constexpr bool val6 = std::is_trivially_move_constructible_v<HalfFp16>;*/

    /*
        bool ok = cpu::Image<Pixel8uC1>::CanLoad(R"(C:\Users\kunz_\OneDrive\Desktop\Unbenannt-3.tif)");
        if (ok)
        {
            auto imgs = cpu::Image<Pixel8uC1>::LoadPlanar(R"(C:\Users\kunz_\OneDrive\Desktop\Unbenannt-1.tif)");

            cpu::Image<Pixel8uC3> img2(imgs[0].SizeAlloc());

            auto it0 = imgs[0].begin();
            auto it1 = imgs[1].begin();
            auto it2 = imgs[2].begin();
            for (auto &elem : img2)
            {
                elem.Value().x = it0->x;
                elem.Value().y = it1->x;
                elem.Value().z = it2->x;
                it0++;
                it1++;
                it2++;
            }

            opp::fileIO::TIFFFile::WriteTIFF(R"(C:\Users\kunz_\OneDrive\Desktop\Unbenannt-3.tif)",
       imgs[0].SizeAlloc().x, imgs[0].SizeAlloc().y, 0.0, PixelTypeEnum::PTE8uC1, imgs[0].Pointer(), imgs[1].Pointer(),
       imgs[2].Pointer(), nullptr, 9);

            img2.Save(R"(C:\Users\kunz_\OneDrive\Desktop\Unbenannt-2.tif)");
        } */
    try
    {

        /*std::cout << "Scalefactor 4 = " << GetScaleFactor(4) << std::endl;
        std::cout << "Scalefactor 2 = " << GetScaleFactor(2) << std::endl;
        std::cout << "Scalefactor 1 = " << GetScaleFactor(1) << std::endl;
        std::cout << "Scalefactor 0 = " << GetScaleFactor(0) << std::endl;
        std::cout << "Scalefactor -1 = " << GetScaleFactor(-1) << std::endl;
        std::cout << "Scalefactor -2 = " << GetScaleFactor(-2) << std::endl;
        std::cout << "Scalefactor -4 = " << GetScaleFactor(-4) << std::endl;*/

#ifdef OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE
        /*auto reg = opp::cuda::GetTemplateInstances();
        for (auto &[functionName, typeInstances] : reg)
        {
            std::cout << "For kernel " << functionName << " (" << typeInstances.size() << "):\n";
            for (const auto &type : typeInstances)
            {
                std::cout << "   <" << type.srcType << ", " << type.computeType << ", " << type.dstType << ">\n";
            }
        }
        std::cout << std::endl; */
#endif

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        const std::filesystem::path baseDir = std::filesystem::path(PROJECT_SOURCE_DIR) / "test/testData";

        auto cpu_flower = cpu::Image<Pixel8uC3>::Load(baseDir / "flower.tif");
        cpu::Image<Pixel8uC3> cpu_addToFlower(cpu_flower.SizeRoi());
        cpu::Image<Pixel8uC3> cpu_res(cpu_flower.SizeRoi());
        cpu::Image<Pixel8uC3> resGPU(cpu_flower.SizeRoi());

        Image<Pixel8uC4> u8test(128, 128);
        u8test.Set({1, 2, 3, 4});
        DevVar<Pixel32fC4> sums(128);
        u8test.SetRoi(Border(-1, 0));

        // InvokeSumSrc(u8test.PointerRoi(), u8test.Pitch(), sums.Pointer(), u8test.SizeRoi(), oppCtx);

        auto error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
        {
        }
        std::vector<Pixel32fC4> sumsh(128);

        sums >> sumsh;

        std::cout << "";

        /*Image<Pixel8uC1> u8test2(128, 128);
        Image<Pixel8uC4> u8test3(128, 128);
        cpu::Image<Pixel8uC4> cpu_test(128, 128);
        cpu::Image<Pixel8uC1> cpu_test2(128, 128);
        cpu::Image<Pixel8uC4> cpu_test3(128, 128);
        cpu::Image<Pixel32fC4> cpu_test4(128, 128);
        byte val[4] = {1, 2, 3, 4};
        cpu_test.Add(Pixel8uC4(val));
        cpu_test.FillRandom();
        cpu_test4.FillRandom();*/

        // u8test.Set(Pixel8uC4(2, 4, 6, 8));
        //// u8test2.Set(Pixel8uC4(1, 3, 5, 7));

        // u8test.Copy(RGBA::G, u8test2);

        //// u8test2.ResetRoi();
        // cpu_test2 << u8test2;

        // cpu_flower.Convert(conv);
        // conv.Mul(2.0f);
        //// conv.Div(2.0f);
        // conv.Convert(cpu_res, RoundingMode::NearestTiesAwayFromZero);

        // const bool isSame = cpu_flower.IsIdentical(cpu_res);

        // cpu_res.Save(baseDir / "convCPU.tif");
        // cpu_addToFlower.Set(Pixel8uC3(40));
        // cpu_res.Set(Pixel8uC3(255));

        Image<Pixel8uC3> gpu_flower(cpu_flower.SizeRoi());
        Image<Pixel8uC3> gpu_addToFlower(cpu_flower.SizeRoi());
        Image<Pixel8uC3> gpu_res(cpu_flower.SizeRoi());
        DevVar<Pixel8uC3> devConst(1);
        Pixel8uC3 hConst(5, 6, 7);
        devConst << hConst;

        cpu_flower >> gpu_flower;
        gpu_addToFlower.Set(Pixel8uC3(40));
        gpu_res.Set(Pixel8uC3(255));

        cpu_flower.SetRoi(-20);
        cpu_addToFlower.SetRoi(-20);
        cpu_res.SetRoi(-20);
        gpu_flower.SetRoi(-20);
        gpu_addToFlower.SetRoi(-20);
        gpu_res.SetRoi(-20);

        Pixel8uC3 constVal(4, 5, 6);
        gpu_flower.Add(gpu_addToFlower, 0);
        gpu_flower.Add(constVal, 0);
        gpu_flower.Add(devConst, gpu_res, 0);
        gpu_res.Div(gpu_flower, -6, opp::RoundingMode::TowardPositiveInfinity);

        cpu_flower.Add(cpu_addToFlower);
        cpu_flower.Add(constVal);
        cpu_flower.Add(hConst, cpu_res);
        cpu_res.Div(cpu_flower, -6, opp::RoundingMode::TowardPositiveInfinity);

        resGPU << gpu_res;

        cpu_res.Save(baseDir / "resCPU.tif");
        resGPU.Save(baseDir / "resGPU.tif");

        cpu_res.ResetRoi();
        resGPU.ResetRoi();
        bool issame = cpu_res.IsIdentical(resGPU);

        std::cout << "Images are identical: " << std::boolalpha << issame << std::endl;

        std::cout << "Done!";
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

// NOLINTEND