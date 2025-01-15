
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

// #include<backends / cuda / image / arithmetic / add.cu>
// #include <backends/cuda/image/arithmetic/add.h>
#include <backends/cuda/image/configurations.h>
// #include <backends/cuda/image/forEachPixelKernel.h>
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

#include <common/image/pixelTypeEnabler.h>
using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;
namespace cpu = opp::image::cpuSimple;

int main()
{
    half_float::half a(-3.14151689f, std::round_to_nearest);
    half_float::half b(-3.14151689f, std::round_toward_zero);
    half_float::half c(-3.14151689f, std::round_toward_infinity);
    half_float::half d(-3.14151689f, std::round_toward_neg_infinity);

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;

    std::cout << -3.14151689f << std::endl;

    float f     = 3.14061f;
    BFloat16 a1 = BFloat16::FromFloat(f);
    BFloat16 b1 = BFloat16::FromFloatTruncate(f);
    BFloat16 c1 = BFloat16::FromFloat(-f);
    BFloat16 d1 = BFloat16::FromFloatTruncate(-f);

    std::cout << a1 << std::endl;
    std::cout << b1 << std::endl;
    std::cout << c1 << std::endl;
    std::cout << d1 << std::endl;

    std::cout << f << std::endl;

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
        std::cout << "Scalefactor 4 = " << GetScaleFactor(4) << std::endl;
        std::cout << "Scalefactor 2 = " << GetScaleFactor(2) << std::endl;
        std::cout << "Scalefactor 1 = " << GetScaleFactor(1) << std::endl;
        std::cout << "Scalefactor 0 = " << GetScaleFactor(0) << std::endl;
        std::cout << "Scalefactor -1 = " << GetScaleFactor(-1) << std::endl;
        std::cout << "Scalefactor -2 = " << GetScaleFactor(-2) << std::endl;
        std::cout << "Scalefactor -4 = " << GetScaleFactor(-4) << std::endl;

#ifdef OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE
        auto reg = opp::cuda::GetTemplateInstances();
        for (auto &[functionName, typeInstances] : reg)
        {
            std::cout << "For kernel " << functionName << " (" << typeInstances.size() << "):\n";
            for (const auto &type : typeInstances)
            {
                std::cout << "   <" << type.srcType << ", " << type.computeType << ", " << type.dstType << ">\n";
            }
        }
        std::cout << std::endl;
#endif

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        const std::filesystem::path baseDir = std::filesystem::path(PROJECT_SOURCE_DIR) / "test/testData";

        auto flower = cpu::Image<Pixel8uC3>::Load(baseDir / "flower.tif");
        cpu::Image<Pixel8uC3> addToFlower(flower.SizeRoi());
        cpu::Image<Pixel8uC3> res(flower.SizeRoi());
        cpu::Image<Pixel8uC3> resGPU(flower.SizeRoi());

        addToFlower.Set(Pixel8uC3(40));
        res.Set(Pixel8uC3(255));

        Image<Pixel8uC3> image1(flower.SizeRoi());
        Image<Pixel8uC3> image2(flower.SizeRoi());
        Image<Pixel8uC3> image3(flower.SizeRoi());
        DevVar<Pixel8uC3> devConst(1);
        Pixel8uC3 hConst(5, 6, 7);
        devConst << hConst;

        flower >> image1;
        image2.Set(Pixel8uC3(40));
        image3.Set(Pixel8uC3(255));

        flower.SetRoi(-20);
        addToFlower.SetRoi(-20);
        res.SetRoi(-20);
        image1.SetRoi(-20);
        image2.SetRoi(-20);
        image3.SetRoi(-20);

        Pixel8uC3 constVal(4, 5, 6);
        image1.Add(image2, 0);
        image1.Add(constVal, 0);
        /*image1.ResetRoi();
        image3.ResetRoi();*/
        image1.Add(devConst, image3, 0);

        addToFlower.Add(constVal);
        addToFlower.Add(hConst);
        flower.Add(addToFlower, res);

        resGPU << image3;

        res.Save(baseDir / "resCPU.tif");
        resGPU.Save(baseDir / "resGPU.tif");

        res.ResetRoi();
        resGPU.ResetRoi();
        bool issame = res.IsIdentical(resGPU);

        std::cout << "Images are identical: " << std::boolalpha << issame << std::endl;

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