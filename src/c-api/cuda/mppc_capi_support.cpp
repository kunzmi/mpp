// deactivate the DLL export macros in imageView:
#include <backends/cuda/image/dllexport_cudai.h>
#undef MPPEXPORT_CUDAI
#define MPPEXPORT_CUDAI
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,modernize-use-using,cppcoreguidelines-pro-type-const-cast,misc-include-cleaner,bugprone-easily-swappable-parameters)
#include "catch_and_return.h"
#include "dllexport.h"
#include <algorithm>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/image/imageView_arithmetic_impl.h>          //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_colorConversion_impl.h>     //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_dataExchangeAndInit_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_filtering_impl.h>           //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_geometryTransforms_impl.h>  //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_morphology_impl.h>          //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_statistics_impl.h>          //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_thresholdAndCompare_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/cuda/streamCtx.h>                                //NOLINT(misc-include-cleaner)
#include <common/colorConversion/colorMatrices.h>
#include <common/errorMessageSingleton.h>
#include <common/exception.h>
#include <common/image/affineTransformation.h>
#include <common/image/bound.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/mpp_defs.h>
#include <common/vectorTypes.h>
#include <driver_types.h>
#include <exception>

#include <common/disableWarningsBegin.h>

#include "mppc_capi_defs.h"

using namespace mpp;
using namespace mpp::image;
using namespace mpp::cuda;
using namespace mpp::image::cuda;

extern "C"
{
    MPPErrorCode DLLEXPORT mppciEvenLevels(Mpp32s *aHPtrLevels, Mpp32s aNumLevels, Mpp32s aLowerLevel,
                                           Mpp32s aUpperLevel, MPPHistorgamEvenMode aHistorgamEvenMode)
    {
        try
        {
            ImageView<Pixel8uC1>::EvenLevels(aHPtrLevels, aNumLevels, aLowerLevel, aUpperLevel,
                                             static_cast<HistorgamEvenMode>(aHistorgamEvenMode));
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciResizeGetNPPShift(Mpp32f aShift[2], MppiSize aSizeSrc, MppiSize aSizeDst)
    {
        try
        {
            checkNullptr(aShift);
            const Size2D sizeSrc(aSizeSrc.width, aSizeSrc.height);
            const Size2D sizeDst(aSizeDst.width, aSizeDst.height);
            const Vec2f scaleFactor    = Vec2f(sizeDst) / Vec2d(sizeSrc);
            const Vec2f invScaleFactor = 1.0f / scaleFactor;
            Vec2f shift(0); // no shift if scaling == 1

            if (scaleFactor.x > 1.0f) // upscaling
            {
                shift.x = (0.25f - (1.0f - invScaleFactor.x) / 2.0f) * scaleFactor.x;
            }
            else if (scaleFactor.x < 1.0f) // downscaling
            {
                shift.x = -((1.0f - invScaleFactor.x) / 2.0f) * scaleFactor.x;
            }

            if (scaleFactor.y > 1.0f) // upscaling
            {
                shift.y = (0.25f - (1.0f - invScaleFactor.y) / 2.0f) * scaleFactor.y;
            }
            else if (scaleFactor.y < 1.0f) // downscaling
            {
                shift.y = -((1.0f - invScaleFactor.y) / 2.0f) * scaleFactor.y;
            }

            aShift[0] = shift.x;
            aShift[1] = shift.y;
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetRotateQuad(MppiRect aSrcROI, double aQuad[4][2], double aAngleInDeg, double aShiftX,
                                              double aShiftY)
    {
        try
        {
            checkNullptr(aQuad);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);

            const AffineTransformation<double> rotate =
                AffineTransformation<double>::GetTranslation(Vec2d(aShiftX, aShiftY)) *
                AffineTransformation<double>::GetRotation(aAngleInDeg);

            const Quad<double> quad = rotate * _SrcRoi;

            aQuad[0][0] = quad.P0.x;
            aQuad[0][1] = quad.P0.y;

            aQuad[1][0] = quad.P1.x;
            aQuad[1][1] = quad.P1.y;

            aQuad[2][0] = quad.P2.x;
            aQuad[2][1] = quad.P2.y;

            aQuad[3][0] = quad.P3.x;
            aQuad[3][1] = quad.P3.y;
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetRotateBound(MppiRect aSrcROI, double aBoundingBox[2][2], double aAngleInDeg,
                                               double aShiftX, double aShiftY)
    {
        try
        {
            checkNullptr(aBoundingBox);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);

            const AffineTransformation<double> rotate =
                AffineTransformation<double>::GetTranslation(Vec2d(aShiftX, aShiftY)) *
                AffineTransformation<double>::GetRotation(aAngleInDeg);

            const Quad<double> quad = rotate * _SrcRoi;
            const Bound<double> bound(quad);

            aBoundingBox[0][0] = bound.Min.x;
            aBoundingBox[0][1] = bound.Min.y;

            aBoundingBox[1][0] = bound.Max.x;
            aBoundingBox[1][1] = bound.Max.y;
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetAffineTransform(MppiRect aSrcROI, const double aQuad[4][2], double aCoeffs[2][3])
    {
        try
        {
            checkNullptr(aQuad);
            checkNullptr(aCoeffs);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);
            const Quad _Quad(aQuad);

            AffineTransformation<double> affine = AffineTransformation<double>::FromQuads(_SrcRoi, _Quad);
            std::copy(affine.Data(), affine.Data() + 6, &aCoeffs[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetAffineQuad(MppiRect aSrcROI, double aQuad[4][2], const double aCoeffs[2][3])
    {
        try
        {
            checkNullptr(aQuad);
            checkNullptr(aCoeffs);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);
            const AffineTransformation<double> affine(aCoeffs);

            const Quad<double> quad = affine * _SrcRoi;

            aQuad[0][0] = quad.P0.x;
            aQuad[0][1] = quad.P0.y;

            aQuad[1][0] = quad.P1.x;
            aQuad[1][1] = quad.P1.y;

            aQuad[2][0] = quad.P2.x;
            aQuad[2][1] = quad.P2.y;

            aQuad[3][0] = quad.P3.x;
            aQuad[3][1] = quad.P3.y;
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetAffineBound(MppiRect aSrcROI, double aBoundingBox[2][2], const double aCoeffs[2][3])
    {
        try
        {
            checkNullptr(aBoundingBox);
            checkNullptr(aCoeffs);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);
            const AffineTransformation<double> affine(aCoeffs);

            const Quad<double> quad = affine * _SrcRoi;
            const Bound<double> bound(quad);

            aBoundingBox[0][0] = bound.Min.x;
            aBoundingBox[0][1] = bound.Min.y;

            aBoundingBox[1][0] = bound.Max.x;
            aBoundingBox[1][1] = bound.Max.y;
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetPerspectiveTransform(MppiRect aSrcROI, const double aQuad[4][2],
                                                        double aCoeffs[3][3])
    {
        try
        {
            checkNullptr(aQuad);
            checkNullptr(aCoeffs);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);
            const Quad _Quad(aQuad);

            const PerspectiveTransformation<double> perspective =
                PerspectiveTransformation<double>::FromQuads(_SrcRoi, _Quad);
            std::copy(perspective.Data(), perspective.Data() + 9, &aCoeffs[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetPerspectiveQuad(MppiRect aSrcROI, double aQuad[4][2], const double aCoeffs[3][3])
    {
        try
        {
            checkNullptr(aQuad);
            checkNullptr(aCoeffs);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);
            const PerspectiveTransformation<double> perspective(aCoeffs);

            const Quad<double> quad = perspective * _SrcRoi;

            aQuad[0][0] = quad.P0.x;
            aQuad[0][1] = quad.P0.y;

            aQuad[1][0] = quad.P1.x;
            aQuad[1][1] = quad.P1.y;

            aQuad[2][0] = quad.P2.x;
            aQuad[2][1] = quad.P2.y;

            aQuad[3][0] = quad.P3.x;
            aQuad[3][1] = quad.P3.y;
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetPerspectiveBound(MppiRect aSrcROI, double aBoundingBox[2][2],
                                                    const double aCoeffs[3][3])
    {
        try
        {
            checkNullptr(aBoundingBox);
            checkNullptr(aCoeffs);
            const Roi _SrcRoi(aSrcROI.x, aSrcROI.y, aSrcROI.width, aSrcROI.height);
            const PerspectiveTransformation<double> perspective(aCoeffs);

            const Quad<double> quad = perspective * _SrcRoi;
            const Bound<double> bound(quad);

            aBoundingBox[0][0] = bound.Min.x;
            aBoundingBox[0][1] = bound.Min.y;

            aBoundingBox[1][0] = bound.Max.x;
            aBoundingBox[1][1] = bound.Max.y;
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoXYZ(float aTwistMatrix[3][3])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoXYZ.Data(), mpp::image::color::RGBtoXYZ.Data() + 9, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoXYZ(float aTwistMatrix[3][3])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoXYZ.Data(), mpp::image::color::BGRtoXYZ.Data() + 9, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixXYZtoRGB(float aTwistMatrix[3][3])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::XYZtoRGB.Data(), mpp::image::color::XYZtoRGB.Data() + 9, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixXYZtoBGR(float aTwistMatrix[3][3])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::XYZtoBGR.Data(), mpp::image::color::XYZtoBGR.Data() + 9, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCC.Data(), mpp::image::color::RGBtoYCC.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCC.Data(), mpp::image::color::BGRtoYCC.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCCtoRGB_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCCtoRGB.Data(), mpp::image::color::YCCtoRGB.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCCtoBGR_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCCtoBGR.Data(), mpp::image::color::YCCtoBGR.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYUV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYUV.Data(), mpp::image::color::RGBtoYUV.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYUV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYUV.Data(), mpp::image::color::BGRtoYUV.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYUVtoRGB_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YUVtoRGB.Data(), mpp::image::color::YUVtoRGB.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYUVtoBGR_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YUVtoBGR.Data(), mpp::image::color::YUVtoBGR.Data() + 12, &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCbCr_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCbCr.Data(), mpp::image::color::RGBtoYCbCr.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCbCr_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCbCr.Data(), mpp::image::color::BGRtoYCbCr.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoRGB_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoRGB.Data(), mpp::image::color::YCbCrtoRGB.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoBGR_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoBGR.Data(), mpp::image::color::YCbCrtoBGR.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCrCb_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCrCb.Data(), mpp::image::color::RGBtoYCrCb.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCrCb_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCrCb.Data(), mpp::image::color::BGRtoYCrCb.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoRGB_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoRGB.Data(), mpp::image::color::YCrCbtoRGB.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoBGR_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoBGR.Data(), mpp::image::color::YCrCbtoBGR.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCbCr_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCbCr_CSC.Data(), mpp::image::color::RGBtoYCbCr_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCbCr_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCbCr_CSC.Data(), mpp::image::color::BGRtoYCbCr_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoRGB_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoRGB_CSC.Data(), mpp::image::color::YCbCrtoRGB_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoBGR_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoBGR_CSC.Data(), mpp::image::color::YCbCrtoBGR_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCrCb_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCrCb_CSC.Data(), mpp::image::color::RGBtoYCrCb_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCrCb_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCrCb_CSC.Data(), mpp::image::color::BGRtoYCrCb_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoRGB_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoRGB_CSC.Data(), mpp::image::color::YCrCbtoRGB_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoBGR_CSC_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoBGR_CSC.Data(), mpp::image::color::YCrCbtoBGR_CSC.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCbCr_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCbCr_JPEG.Data(), mpp::image::color::RGBtoYCbCr_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCbCr_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCbCr_JPEG.Data(), mpp::image::color::BGRtoYCbCr_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoRGB_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoRGB_JPEG.Data(), mpp::image::color::YCbCrtoRGB_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoBGR_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoBGR_JPEG.Data(), mpp::image::color::YCbCrtoBGR_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCrCb_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCrCb_JPEG.Data(), mpp::image::color::RGBtoYCrCb_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCrCb_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCrCb_JPEG.Data(), mpp::image::color::BGRtoYCrCb_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoRGB_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoRGB_JPEG.Data(), mpp::image::color::YCrCbtoRGB_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoBGR_JPEG_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoBGR_JPEG.Data(), mpp::image::color::YCrCbtoBGR_JPEG.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCbCr_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCbCr_HDTV.Data(), mpp::image::color::RGBtoYCbCr_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCbCr_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCbCr_HDTV.Data(), mpp::image::color::BGRtoYCbCr_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoRGB_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoRGB_HDTV.Data(), mpp::image::color::YCbCrtoRGB_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCbCrtoBGR_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCbCrtoBGR_HDTV.Data(), mpp::image::color::YCbCrtoBGR_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixRGBtoYCrCb_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::RGBtoYCrCb_HDTV.Data(), mpp::image::color::RGBtoYCrCb_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixBGRtoYCrCb_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::BGRtoYCrCb_HDTV.Data(), mpp::image::color::BGRtoYCrCb_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoRGB_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoRGB_HDTV.Data(), mpp::image::color::YCrCbtoRGB_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciGetMatrixYCrCbtoBGR_HDTV_8u(float aTwistMatrix[3][4])
    {
        try
        {
            checkNullptr(aTwistMatrix);
            std::copy(mpp::image::color::YCrCbtoBGR_HDTV.Data(), mpp::image::color::YCrCbtoBGR_HDTV.Data() + 12,
                      &aTwistMatrix[0][0]);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppcGetLastErrorCode()
    {
        return static_cast<MPPErrorCode>(ErrorMessageSingleton::GetLastErrorCode());
    }

    DLLEXPORT const char *mppcGetLastErrorMessage()
    {
        return ErrorMessageSingleton::GetLastErrorMessage().c_str();
    }

    MPPErrorCode DLLEXPORT mppcGetStreamContext(MppCudaStreamCtx *aStreamCtx)
    {
        try
        {
            checkNullptr(aStreamCtx);

            memcpy(aStreamCtx, &mpp::cuda::StreamCtxSingleton::Get(), sizeof(MppCudaStreamCtx));
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppcSetDefaultStream(cudaStream_t aStream)
    {
        try
        {
            mpp::cuda::StreamCtxSingleton::SetStream(mpp::cuda::Stream(aStream));
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppcUpdateContext()
    {
        try
        {
            mpp::cuda::StreamCtxSingleton::UpdateContext();
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

} // extern "C"
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,modernize-use-using,cppcoreguidelines-pro-type-const-cast,misc-include-cleaner,bugprone-easily-swappable-parameters)
#include <common/disableWarningsEnd.h>
