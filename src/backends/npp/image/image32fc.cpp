#include "image32fc.h"
#include "image32fcC1View.h"
#include "image32fcC2View.h"
#include "image32fcC3View.h"
#include "image32fcC4View.h"
#include <backends/npp/nppException.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <memory>
#include <nppdefs.h>
#include <nppi_support_functions.h>

using namespace mpp::cuda;

namespace mpp::image::npp
{

Image32fcC1::Image32fcC1(int aWidth, int aHeight) : Image32fcC1(Size2D(aWidth, aHeight))
{
}
Image32fcC1::Image32fcC1(const Size2D &aSize) : Image32fcC1View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32fcC1 *>(nppiMalloc_32fc_C1(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32fcC1 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32fcC1::~Image32fcC1()
{
    if (PointerRef() != nullptr)
    {
        nppiFree(Pointer());
    }
    PointerRef()    = nullptr;
    PointerRoiRef() = nullptr;
    PitchRef()      = 0;
    ROIRef()        = Roi();
    SizeAllocRef()  = Size2D();
}

Image32fcC1::Image32fcC1(Image32fcC1 &&aOther) noexcept
{
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();
}

Image32fcC1 &Image32fcC1::operator=(Image32fcC1 &&aOther) noexcept
{
    if (std::addressof(aOther) == std::addressof(*this))
    {
        return *this;
    }
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();

    return *this;
}

Image32fcC2::Image32fcC2(int aWidth, int aHeight) : Image32fcC2(Size2D(aWidth, aHeight))
{
}
Image32fcC2::Image32fcC2(const Size2D &aSize) : Image32fcC2View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32fcC2 *>(nppiMalloc_32fc_C2(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32fcC2 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32fcC2::~Image32fcC2()
{
    if (PointerRef() != nullptr)
    {
        nppiFree(Pointer());
    }
    PointerRef()    = nullptr;
    PointerRoiRef() = nullptr;
    PitchRef()      = 0;
    ROIRef()        = Roi();
    SizeAllocRef()  = Size2D();
}

Image32fcC2::Image32fcC2(Image32fcC2 &&aOther) noexcept
{
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();
}

Image32fcC2 &Image32fcC2::operator=(Image32fcC2 &&aOther) noexcept
{
    if (std::addressof(aOther) == std::addressof(*this))
    {
        return *this;
    }
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();

    return *this;
}

Image32fcC3::Image32fcC3(int aWidth, int aHeight) : Image32fcC3(Size2D(aWidth, aHeight))
{
}
Image32fcC3::Image32fcC3(const Size2D &aSize) : Image32fcC3View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32fcC3 *>(nppiMalloc_32fc_C3(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32fcC3 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32fcC3::~Image32fcC3()
{
    if (PointerRef() != nullptr)
    {
        nppiFree(Pointer());
    }
    PointerRef()    = nullptr;
    PointerRoiRef() = nullptr;
    PitchRef()      = 0;
    ROIRef()        = Roi();
    SizeAllocRef()  = Size2D();
}

Image32fcC3::Image32fcC3(Image32fcC3 &&aOther) noexcept
{
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();
}

Image32fcC3 &Image32fcC3::operator=(Image32fcC3 &&aOther) noexcept
{
    if (std::addressof(aOther) == std::addressof(*this))
    {
        return *this;
    }
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();

    return *this;
}

Image32fcC4::Image32fcC4(int aWidth, int aHeight) : Image32fcC4(Size2D(aWidth, aHeight))
{
}
Image32fcC4::Image32fcC4(const Size2D &aSize) : Image32fcC4View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32fcC4 *>(nppiMalloc_32fc_C4(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32fcC4 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32fcC4::~Image32fcC4()
{
    if (PointerRef() != nullptr)
    {
        nppiFree(Pointer());
    }
    PointerRef()    = nullptr;
    PointerRoiRef() = nullptr;
    PitchRef()      = 0;
    ROIRef()        = Roi();
    SizeAllocRef()  = Size2D();
}

Image32fcC4::Image32fcC4(Image32fcC4 &&aOther) noexcept
{
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();
}

Image32fcC4 &Image32fcC4::operator=(Image32fcC4 &&aOther) noexcept
{
    if (std::addressof(aOther) == std::addressof(*this))
    {
        return *this;
    }
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();

    return *this;
}

} // namespace mpp::image::npp
