
// NOLINTBEGIN

// #include <common/image/size2D.h>
#include <common/safeCast.h>
#include <common/version.h>
#include <cstddef>
// #include <cuda_runtime_api.h>
// #include <driver_types.h>
#include <common/opp_defs.h>
#include <iostream>
#include <vector>

#include <common/arithmetic/binary_operators.h>
// #include <common/arithmetic/ternary_operators.h>
// #include <common/arithmetic/unary_operators.h>
// #include <common/image/pixelTypes.h>
// #include <common/vector_typetraits.h>
// #include <common/vector4A.h>
// #include <common/vectorTypes.h>

#include <backends/cuda/image/addKernel.h>
#include <backends/cuda/image/configurations.h>
//
#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <backends/simple_cpu/image/forEachPixelMasked_impl.h>
#include <backends/simple_cpu/image/forEachPixel_impl.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/half_fp16.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/vector1.h>
#include <common/vector2.h>
#include <common/vectorTypes.h>
// #include <common/scratchBuffer.h>
// #include <half/half.hpp>

using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;
namespace cpu = opp::image::cpuSimple;

void fun(Pixel8uC4 aTest)
{
    std::cout << "Test is: " << aTest << std::endl;
}

int main()
{
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
#ifdef OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE
        auto reg = opp::cuda::GetTemplateInstances();
        for (auto &elem : reg)
        {
            std::cout << elem << std::endl;
        }
#endif
        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        /*DevVar<byte> buffer(1024);
        std::vector<byte> hBuffer(1024);
        DevVarView<byte> buffer2(buffer);
        buffer << hBuffer;*/

        const std::filesystem::path baseDir = std::filesystem::path(PROJECT_SOURCE_DIR) / "test/testData";

        auto flower = cpu::Image<Pixel8uC3>::Load(baseDir / "flower.tif");
        cpu::Image<Pixel8uC3> addToFlower(flower.SizeRoi());
        cpu::Image<Pixel8uC3> res(flower.SizeRoi());
        cpu::Image<Pixel8uC3> resGPU(flower.SizeRoi());

        for (auto &pixelIterator : addToFlower)
        {
            pixelIterator.Value() = 40;
        }
        for (auto &pixelIterator : res)
        {
            pixelIterator.Value() = 255;
        }
        for (auto &pixelIterator : resGPU)
        {
            pixelIterator.Value() = 255;
        }

        Image<Pixel8uC3> image1(flower.SizeRoi());
        Image<Pixel8uC3> image2(flower.SizeRoi());
        Image<Pixel8uC3> image3(flower.SizeRoi());

        flower >> image1;
        addToFlower >> image2;
        resGPU >> image3;

        flower.SetRoi(-20);
        addToFlower.SetRoi(-20);
        res.SetRoi(-20);
        image1.SetRoi(-20);
        image2.SetRoi(-20);
        image3.SetRoi(-20);

        opp::cuda::StreamCtx ctx = opp::cuda::StreamCtxSingleton::Get();

        InvokeAddSrcSrc(image1.PointerRoi(), image1.Pitch(), image2.PointerRoi(), image2.Pitch(), image3.PointerRoi(),
                        image3.Pitch(), image1.SizeRoi(), ctx);

        flower.Add(addToFlower, res);

        resGPU << image3;

        res.Save(baseDir / "resCPU.tif");
        resGPU.Save(baseDir / "resGPU.tif");

        res.ResetRoi();
        resGPU.ResetRoi();
        bool issame = res.IsIdentical(resGPU);

        std::cout << "Images are identical: " << (issame ? "true" : "false") << std::endl;

        /*InvokeAddSrcSrc(image1.PointerRoi(), image1.Pitch(), image2.PointerRoi(), image2.Pitch(), image3.PointerRoi(),
                        image3.Pitch(), roi.Size(), ctx);

        InvokeAddSrcSrc<Pixel8uC4, Pixel8uC4, Pixel8uC4>(image4.PointerRoi(), image4.Pitch(), image5.PointerRoi(),
                                                         image5.Pitch(), image6.PointerRoi(), image6.Pitch(),
                                                         roi.Size(), ctx);*/

        std::cout << "Done!";
    }
    catch (const opp::OPPException &ex)
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