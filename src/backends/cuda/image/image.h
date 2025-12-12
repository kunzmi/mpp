#pragma once
#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <vector>

namespace mpp::image::cuda
{

template <PixelType T> class Image : public ImageView<T>
{

  public:
    Image() = delete;
    Image(int aWidth, int aHeight) : Image(Size2D(aWidth, aHeight))
    {
    }
    Image(const Size2D &aSize) : ImageView<T>(aSize)
    {
        // NPP seems to use a minimum pitch of 512 bytes, whereas cudaMallocPitch seems to be 256 bytes...
        // As for now I don't really see any reason why 512 are "better" than 256, so stick to cudaMallocPitch:
        cudaSafeCall(cudaMallocPitch(reinterpret_cast<void **>(&ImageView<T>::PointerRef()), &ImageView<T>::PitchRef(),
                                     sizeof(T) * to_size_t(aSize.x), to_size_t(aSize.y)));
        ImageView<T>::PointerRoiRef() = ImageView<T>::PointerRef();
    }
    ~Image() override
    {
        if (ImageView<T>::PointerRef() != nullptr)
        {
            cudaSafeCall(cudaFree(ImageView<T>::PointerRef()));
        }
        ImageView<T>::PointerRef()    = nullptr;
        ImageView<T>::PointerRoiRef() = nullptr;
        ImageView<T>::PitchRef()      = 0;
        ImageView<T>::ROIRef()        = Roi();
        ImageView<T>::SizeAllocRef()  = Size2D();
    }

    Image(const Image &) = delete;
    Image(Image &&aOther) noexcept
    {
        ImageView<T>::PointerRef()    = aOther.PointerRef();
        ImageView<T>::PointerRoiRef() = aOther.PointerRoiRef();
        ImageView<T>::PitchRef()      = aOther.PitchRef();
        ImageView<T>::ROIRef()        = aOther.ROIRef();
        ImageView<T>::SizeAllocRef()  = aOther.SizeAllocRef();

        aOther.PointerRef()    = nullptr;
        aOther.PointerRoiRef() = nullptr;
        aOther.PitchRef()      = 0;
        aOther.ROIRef()        = Roi();
        aOther.SizeAllocRef()  = Size2D();
    }

    Image &operator=(const Image &) = delete;
    Image &operator=(Image &&aOther) noexcept
    {
        if (std::addressof(aOther) == std::addressof(*this))
        {
            return *this;
        }
        ImageView<T>::PointerRef()    = aOther.PointerRef();
        ImageView<T>::PointerRoiRef() = aOther.PointerRoiRef();
        ImageView<T>::PitchRef()      = aOther.PitchRef();
        ImageView<T>::ROIRef()        = aOther.ROIRef();
        ImageView<T>::SizeAllocRef()  = aOther.SizeAllocRef();

        aOther.PointerRef()    = nullptr;
        aOther.PointerRoiRef() = nullptr;
        aOther.PitchRef()      = 0;
        aOther.ROIRef()        = Roi();
        aOther.SizeAllocRef()  = Size2D();
        return *this;
    }
};
} // namespace mpp::image::cuda