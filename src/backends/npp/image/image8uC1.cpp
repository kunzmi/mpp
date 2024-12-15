#include "image8uC1.h"
#include "image8uC1View.h"
#include <backends/cuda/devVarView.h>
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

using namespace opp::cuda;

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
void Image8uC1::Set(const Npp8u nValue, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiSet_8u_C1R_Ctx(nValue, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Dst: " << ROI() );
}

void Image8uC1::Set(const Pixel8uC1 &nValue, const Image8uC1 &pMask, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiSet_8u_C1MR_Ctx(nValue.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), reinterpret_cast<const Npp8u *>(pMask.PointerRoi()), to_int(pMask.Pitch()), nppStreamCtx), 
                   "ROI Dst: " << ROI() );
}

void Image8uC1::Copy(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopy_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Copy(const Image8uC1 &pSrc, const Image8uC1 &pMask, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopy_8u_C1MR_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), reinterpret_cast<const Npp8u *>(pMask.PointerRoi()), to_int(pMask.Pitch()), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Copy(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopy_8u_C3C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Copy(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopy_8u_C4C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::CopyConstBorder(const Image8uC1 &pSrc, int nTopBorderHeight, int nLeftBorderWidth, const Pixel8uC1 &nValue, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopyConstBorder_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), pSrc.NppiSizeRoi(), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nTopBorderHeight, nLeftBorderWidth, nValue.x, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::CopyReplicateBorder(const Image8uC1 &pSrc, int nTopBorderHeight, int nLeftBorderWidth, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopyReplicateBorder_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), pSrc.NppiSizeRoi(), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nTopBorderHeight, nLeftBorderWidth, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::CopyWrapBorder(const Image8uC1 &pSrc, int nTopBorderHeight, int nLeftBorderWidth, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopyWrapBorder_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), pSrc.NppiSizeRoi(), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nTopBorderHeight, nLeftBorderWidth, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::CopySubpix(const Image8uC1 &pSrc, Npp32f nDx, Npp32f nDy, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiCopySubpix_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nDx, nDy, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Transpose(const Image8uC1 &pSrc, NppiSize oSrcROI, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiTranspose_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), oSrcROI, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Add(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiAddC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Add(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiAddDeviceC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Add(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiAddC_8u_C1IRSfs_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Add(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiAddDeviceC_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Mul(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiMulC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Mul(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiMulDeviceC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Mul(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiMulC_8u_C1IRSfs_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Mul(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiMulDeviceC_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::MulScale(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiMulCScale_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::MulScale(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiMulDeviceCScale_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::MulScale(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiMulCScale_8u_C1IR_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::MulScale(const cuda::DevVarView<Pixel8uC1> &pConstant, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiMulDeviceCScale_8u_C1IR_Ctx(reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Sub(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiSubC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Sub(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiSubDeviceC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Sub(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiSubC_8u_C1IRSfs_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Sub(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiSubDeviceC_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Div(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiDivC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Div(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiDivDeviceC_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Div(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiDivC_8u_C1IRSfs_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Div(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiDivDeviceC_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pConstant.Pointer()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::AbsDiff(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiAbsDiffC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nConstant.x, nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::AbsDiff(const Image8uC1 &pSrc1, cuda::DevVarView<Pixel8uC1> &pConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiAbsDiffDeviceC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), reinterpret_cast<Npp8u *>(pConstant.Pointer()), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Add(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiAdd_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Add(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiAdd_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::Mul(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiMul_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Mul(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiMul_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::MulScale(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiMulScale_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::MulScale(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiMulScale_8u_C1IR_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::Sub(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiSub_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Sub(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiSub_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::Div(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiDiv_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Div(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiDiv_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::Div_Round(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, NppRoundMode rndMode, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiDiv_Round_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), rndMode, nScaleFactor, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Div_Round(const Image8uC1 &pSrc, NppRoundMode rndMode, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiDiv_Round_8u_C1IRSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), rndMode, nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::AbsDiff(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiAbsDiff_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Sqr(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiSqr_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Sqr(int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiSqr_8u_C1IRSfs_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Sqrt(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiSqrt_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Sqrt(int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiSqrt_8u_C1IRSfs_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Ln(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiLn_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Ln(int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiLn_8u_C1IRSfs_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Exp(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiExp_8u_C1RSfs_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Exp(int nScaleFactor, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiExp_8u_C1IRSfs_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nScaleFactor, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::And(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiAndC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::And(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiAndC_8u_C1IR_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Or(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiOrC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Or(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiOrC_8u_C1IR_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::Xor(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiXorC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Xor(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiXorC_8u_C1IR_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::RShift(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiRShiftC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::RShift(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiRShiftC_8u_C1IR_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::LShift(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiLShiftC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::LShift(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiLShiftC_8u_C1IR_Ctx(nConstant.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::And(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiAnd_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::And(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiAnd_8u_C1IR_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::Or(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiOr_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Or(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiOr_8u_C1IR_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::Xor(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiXor_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Xor(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiXor_8u_C1IR_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI SrcDst: " << ROI() );
}

void Image8uC1::Not(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiNot_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::Not(const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiNot_8u_C1IR_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::AlphaComp(const Image8uC1 &pSrc1, const Pixel8uC1 &nAlpha1, const Image8uC1 &pSrc2, const Pixel8uC1 &nAlpha2, NppiAlphaOp eAlphaOp, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiAlphaCompC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nAlpha1.x, reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), nAlpha2.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), eAlphaOp, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::AlphaPremul(const Image8uC1 &pSrc1, const Pixel8uC1 &nAlpha1, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    nppSafeCallExt(nppiAlphaPremulC_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), nAlpha1.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc1.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::AlphaPremul(const Pixel8uC1 &nAlpha1, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiAlphaPremulC_8u_C1IR_Ctx(nAlpha1.x, reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::AlphaComp(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, NppiAlphaOp eAlphaOp, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiAlphaComp_8u_AC1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), eAlphaOp, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::RGBToGray(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiRGBToGray_8u_C3C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::RGBToGray(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiRGBToGray_8u_AC4C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::ColorToGray(const Image8uC1 &pSrc, const Npp32f[3] aCoeffs, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiColorToGray_8u_C3C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), aCoeffs, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::ColorToGray(const Image8uC1 &pSrc, const Npp32f[3] aCoeffs, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiColorToGray_8u_AC4C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), aCoeffs, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::ColorToGray(const Image8uC1 &pSrc, const Npp32f[4] aCoeffs, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiColorToGray_8u_C4C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), aCoeffs, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::GradientColorToGray(const Image8uC1 &pSrc, NppiNorm eNorm, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiGradientColorToGray_8u_C3C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), eNorm, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::CompColorKey(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const Pixel8uC1 &nColorKeyConst, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc1.ROI());
    checkSameSize(ROI(), pSrc2.ROI());
    nppSafeCallExt(nppiCompColorKey_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc1.PointerRoi()), to_int(pSrc1.Pitch()), reinterpret_cast<const Npp8u *>(pSrc2.PointerRoi()), to_int(pSrc2.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), nColorKeyConst.x, nppStreamCtx), 
                   "ROI Src1: " << pSrc1.ROI() << " ROI Src2: " << pSrc2.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::ColorTwist32f(const Image8uC1 &pSrc, const Npp32f[3][4] aTwist, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiColorTwist32f_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), aTwist, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::ColorTwist32f(const Npp32f[3][4] aTwist, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiColorTwist32f_8u_C1IR_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), aTwist, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::ColorTwistBatch32f(Npp32f nMin, Npp32f nMax, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiColorTwistBatch32f_8u_C1R_Ctx(nMin, nMax, NppiSizeRoi(), pBatchList, nBatchSize, nppStreamCtx), 
                   "ROI Dst: " << ROI() );
}

void Image8uC1::ColorTwistBatch32f(Npp32f nMin, Npp32f nMax, NppiColorTwistBatchCXR * pBatchList, int nBatchSize, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiColorTwistBatch32f_8u_C1IR_Ctx(nMin, nMax, NppiSizeRoi(), pBatchList, nBatchSize, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::LUT(const Image8uC1 &pSrc, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiLUT_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), pValues, pLevels, nLevels, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::LUT(const Npp32s * pValues, const Npp32s * pLevels, int nLevels, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiLUT_8u_C1IR_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), pValues, pLevels, nLevels, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::LUT_Linear(const Image8uC1 &pSrc, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiLUT_Linear_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), pValues, pLevels, nLevels, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::LUT_Linear(const Npp32s * pValues, const Npp32s * pLevels, int nLevels, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiLUT_Linear_8u_C1IR_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), pValues, pLevels, nLevels, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::LUT_Cubic(const Image8uC1 &pSrc, const Npp32s * pValues, const Npp32s * pLevels, int nLevels, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiLUT_Cubic_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), pValues, pLevels, nLevels, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}

void Image8uC1::LUT_Cubic(const Npp32s * pValues, const Npp32s * pLevels, int nLevels, const NppStreamContext &nppStreamCtx)
{
    nppSafeCallExt(nppiLUT_Cubic_8u_C1IR_Ctx(reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), pValues, pLevels, nLevels, nppStreamCtx), 
                   "ROI SrcDst: " << ROI() );
}

void Image8uC1::LUTPalette(const Image8uC1 &pSrc, const Npp8u * pTable, int nBitSize, const NppStreamContext &nppStreamCtx)
{
    checkSameSize(ROI(), pSrc.ROI());
    nppSafeCallExt(nppiLUTPalette_8u_C1R_Ctx(reinterpret_cast<const Npp8u *>(pSrc.PointerRoi()), to_int(pSrc.Pitch()), reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch()), NppiSizeRoi(), pTable, nBitSize, nppStreamCtx), 
                   "ROI Src: " << pSrc.ROI() << " ROI Dst: " << ROI() );
}



} // namespace opp::image::npp
