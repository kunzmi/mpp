#include "image8uC1.h"
#include "image8uC1View.h"
#include <backends/npp/nppException.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <memory>
#include <nppdefs.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_filtering_functions.h>
#include <nppi_geometry_transforms.h>
#include <nppi_linear_transforms.h>
#include <nppi_morphological_operations.h>
#include <nppi_statistics_functions.h>
#include <nppi_support_functions.h>
#include <nppi_threshold_and_compare_operations.h>

namespace opp::image::npp
{

Image8uC1::Image8uC1(int aWidth, int aHeight) : Image8uC1(Size2D(aWidth, aHeight))
{
}
Image8uC1::Image8uC1(const Size2D &aSize) : Image8uC1View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel8uC1 *>(nppiMalloc_8u_C1(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image8uC1 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image8uC1::~Image8uC1()
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

Image8uC1::Image8uC1(Image8uC1 &&aOther) noexcept
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

Image8uC1 &Image8uC1::operator=(Image8uC1 &&aOther) noexcept
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
void Image8uC1::Add(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx)
{
    checkSameSize(ROI(), aSrc1.ROI());
    checkSameSize(ROI(), aSrc2.ROI());
    nppSafeCallExt(nppiAdd_8u_C1RSfs_Ctx(aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), PointerRoi(), Pitch(), NppiSize(), aScaleFactor, aStreamCtx);
}


void Image8uC1::Div(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx)
{
    checkSameSize(ROI(), aSrc1.ROI());
    checkSameSize(ROI(), aSrc2.ROI());
    nppSafeCallExt(nppiDiv_8u_C1RSfs_Ctx(aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), PointerRoi(), Pitch(), NppiSize(), aScaleFactor, aStreamCtx);
}


void Image8uC1::Mul(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx)
{
    checkSameSize(ROI(), aSrc1.ROI());
    checkSameSize(ROI(), aSrc2.ROI());
    nppSafeCallExt(nppiMul_8u_C1RSfs_Ctx(aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), PointerRoi(), Pitch(), NppiSize(), aScaleFactor, aStreamCtx);
}


void Image8uC1::Sub(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx)
{
    checkSameSize(ROI(), aSrc1.ROI());
    checkSameSize(ROI(), aSrc2.ROI());
    nppSafeCallExt(nppiSub_8u_C1RSfs_Ctx(aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), PointerRoi(), Pitch(), NppiSize(), aScaleFactor, aStreamCtx);
}




} // namespace opp::image::npp
