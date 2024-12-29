
// NOLINTBEGIN

#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <common/version.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector>

#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/ternary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/image/pixelTypes.h>
#include <common/vector_typetraits.h>
#include <common/vector4A.h>
#include <common/vectorTypes.h>

#include <backends/cuda/image/addKernel.h>
#include <backends/cuda/image/configurations.h>

#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>

#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/scratchBuffer.h>
#include <half/half.hpp>

using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;
namespace cpu = opp::image::cpuSimple;

int main()
{
    Vector1<float> t1(12.0f);
    Vector1<float> t2(13.0f);
    Vector1<byte> t3 = Vector1<float>::CompareLE(t1, t2);

    Vector4<int> i4;
    Vector4<int>::same_vector_size_different_type_t<byte> b4;
    b4.x;
    Vector4<bool> isthisallowed(true, false, false, true);

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

        opp::fileIO::TIFFFile::WriteTIFF(R"(C:\Users\kunz_\OneDrive\Desktop\Unbenannt-3.tif)", imgs[0].SizeAlloc().x,
                                         imgs[0].SizeAlloc().y, 0.0, PixelTypeEnum::PTE8uC1, imgs[0].Pointer(),
                                         imgs[1].Pointer(), imgs[2].Pointer(), nullptr, 9);

        img2.Save(R"(C:\Users\kunz_\OneDrive\Desktop\Unbenannt-2.tif)");
    } /**/

    try
    {
#ifdef OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE
        auto reg = opp::cuda::GetTemplateInstances();
        for (auto &elem : reg)
        {
            std::cout << elem << std::endl;
        }
#endif

        DevVar<byte> buffer(1024);
        std::vector<byte> hBuffer(1024);
        DevVarView<byte> buffer2(buffer);
        buffer << hBuffer;

        Image<Pixel8uC1> image1(1024, 1024);
        Image<Pixel8uC1> image2(1024, 1024);
        Image<Pixel8uC1> image3(1024, 1024);

        Image<Pixel8uC4> image4(1024, 1024);
        Image<Pixel8uC4> image5(1024, 1024);
        Image<Pixel8uC4> image6(1024, 1024);

        std::vector<Pixel8uC1> host1(1024 * 1024, Pixel8uC1(25));
        std::vector<Pixel8uC1> host2(1024 * 1024, Pixel8uC1(10));

        image1 << host1;
        image2 << host2;
        image3 << host2;

        opp::cuda::StreamCtx ctx = opp::cuda::StreamCtxSingleton::Get();
        std::cout << ctx.DeviceId;

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        Roi roi = image1.ROI();
        roi -= 1;
        image1.SetRoi(roi);
        image2.SetRoi(roi);
        image3.SetRoi(roi);

        InvokeAddSrcSrc(image1.PointerRoi(), image1.Pitch(), image2.PointerRoi(), image2.Pitch(), image3.PointerRoi(),
                        image3.Pitch(), roi.Size(), ctx);

        InvokeAddSrcSrc<Pixel8uC4, Pixel8uC4, Pixel8uC4>(image4.PointerRoi(), image4.Pitch(), image5.PointerRoi(),
                                                         image5.Pitch(), image6.PointerRoi(), image6.Pitch(),
                                                         roi.Size(), ctx);

        image3.ResetRoi();
        image3 >> host2;

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