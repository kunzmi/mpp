#include "image32sc.h"
#include "image32scC1View.h"
#include "image32scC2View.h"
#include "image32scC3View.h"
#include "image32scC4View.h"
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

Image32scC1::Image32scC1(int aWidth, int aHeight) : Image32scC1(Size2D(aWidth, aHeight))
{
}
Image32scC1::Image32scC1(const Size2D &aSize) : Image32scC1View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32scC1 *>(nppiMalloc_32sc_C1(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32scC1 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32scC1::~Image32scC1()
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

Image32scC1::Image32scC1(Image32scC1 &&aOther) noexcept
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

Image32scC1 &Image32scC1::operator=(Image32scC1 &&aOther) noexcept
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

Image32scC2::Image32scC2(int aWidth, int aHeight) : Image32scC2(Size2D(aWidth, aHeight))
{
}
Image32scC2::Image32scC2(const Size2D &aSize) : Image32scC2View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32scC2 *>(nppiMalloc_32sc_C2(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32scC2 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32scC2::~Image32scC2()
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

Image32scC2::Image32scC2(Image32scC2 &&aOther) noexcept
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

Image32scC2 &Image32scC2::operator=(Image32scC2 &&aOther) noexcept
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

Image32scC3::Image32scC3(int aWidth, int aHeight) : Image32scC3(Size2D(aWidth, aHeight))
{
}
Image32scC3::Image32scC3(const Size2D &aSize) : Image32scC3View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32scC3 *>(nppiMalloc_32sc_C3(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32scC3 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32scC3::~Image32scC3()
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

Image32scC3::Image32scC3(Image32scC3 &&aOther) noexcept
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

Image32scC3 &Image32scC3::operator=(Image32scC3 &&aOther) noexcept
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

Image32scC4::Image32scC4(int aWidth, int aHeight) : Image32scC4(Size2D(aWidth, aHeight))
{
}
Image32scC4::Image32scC4(const Size2D &aSize) : Image32scC4View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel32scC4 *>(nppiMalloc_32sc_C4(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image32scC4 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image32scC4::~Image32scC4()
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

Image32scC4::Image32scC4(Image32scC4 &&aOther) noexcept
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

Image32scC4 &Image32scC4::operator=(Image32scC4 &&aOther) noexcept
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
