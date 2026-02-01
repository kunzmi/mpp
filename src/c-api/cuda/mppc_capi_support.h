#ifndef MPPI_CUDA_CAPI_SUPPORT_H
#define MPPI_CUDA_CAPI_SUPPORT_H

#include "mppc_capi_defs.h"
#include <driver_types.h> // for cudaStream_t

#ifdef __cplusplus
extern "C"
{
#endif

    /// <summary>
    /// Compute levels with even distribution, depending on aHistorgamEvenMode, this function tries to give identical
    /// results as the same function in NPP or as the methods used in the CUB backend used by MPP for histogram
    /// computation.
    /// </summary>
    /// <param name="aHPtrLevels">A host pointer to array which receives the levels being computed.
    /// The array needs to be of size aNumLevels.</ param>
    /// <param name="aNumLevels">The number of levels being computed. aNumLevels must be at least 2</param>
    /// <param name="aLowerLevel">Lower boundary value of the lowest level.</param>
    /// <param name="aUpperLevel">Upper boundary value of the greatest level.</param>
    /// <param name="aHistorgamEvenMode">Switch compatibility mode: CUB (default) or NPP.</param>
    MPPErrorCode mppciEvenLevels(Mpp32s *aHPtrLevels, Mpp32s aNumLevels, Mpp32s aLowerLevel, Mpp32s aUpperLevel,
                                 MPPHistorgamEvenMode aHistorgamEvenMode);

    /// <summary>
    /// Returns a shift to be used in mppciResizeSqrPixel function that matches the result given by the NPP
    /// Resize-function.
    /// </summary>
    MPPErrorCode mppciResizeGetNPPShift(Mpp32f aShift[2], MppiSize aSizeSrc, MppiSize aSizeDst);

    /// <summary>
    /// Compute shape of rotated image.
    /// </summary>
    /// <param name="aSrcROI">Region-of-interest of the source image.</ param>
    /// <param name="aQuad">Array of 2D points. These points are the locations
    /// of the corners of the rotated ROI.</ param>
    /// <param name="aAngleInDeg">The rotation angle.</param>
    /// <param name="aShiftX">Post-rotation shift in x-direction</ param>
    /// <param name="aShiftY">Post-rotation shift in y-direction</param>
    MPPErrorCode mppciGetRotateQuad(MppiRect aSrcROI, double aQuad[4][2], double aAngleInDeg, double aShiftX,
                                    double aShiftY);

    /// <summary>
    /// Compute bounding-box of rotated image.
    /// </summary>
    /// <param name="aSrcROI">Region-of-interest of the source image.</param>
    /// <param name="aBoundingBox">Two 2D points representing the bounding-box of the rotated image.
    /// All four points from mppciGetRotateQuad are contained inside the axis-aligned rectangle spanned
    /// by the the two *points of this bounding box.</ param>
    /// <param name="aAngleInDeg">The rotation angle.</param>
    /// <param name="aShiftX">Post-rotation shift in x-direction</ param>
    /// <param name="aShiftY">Post-rotation shift in y-direction</param>
    MPPErrorCode mppciGetRotateBound(MppiRect aSrcROI, double aBoundingBox[2][2], double aAngleInDeg, double aShiftX,
                                     double aShiftY);

    /// <summary>
    /// Computes affine transform coefficients based on source ROI and destination quadrilateral.
    /// <para/>
    /// The function computes the coefficients of an affine transformation that maps the
    /// given source ROI (axis aligned rectangle with integer coordinates) to a quadrilateral
    /// in the destination image.
    /// <para/>
    /// An affine transform in 2D is fully determined by the mapping of just three vertices.
    /// This function's API allows for passing a complete quadrilateral effectively making the
    /// prolem overdetermined. What this means in practice is, that for certain quadrilaterals it is
    /// not possible to find an affine transform that would map all four corners of the source
    /// ROI to the four vertices of that quadrilateral.
    /// <para/>
    /// The function circumvents this problem by only looking at the first three vertices of
    /// the destination image quadrilateral to determine the affine transformation's coefficients.
    /// </summary>
    /// <param name="aSrcROI">The source ROI.</param>
    /// <param name="aQuad">The destination quadrilateral.</param>
    /// <param name="aCoeffs">The resulting affine transform coefficients.</param>
    MPPErrorCode mppciGetAffineTransform(MppiRect aSrcROI, const double aQuad[4][2], double aCoeffs[2][3]);

    /// <summary>
    /// Compute shape of transformed image.
    /// <para/>
    /// This method computes the quadrilateral in the destination image that
    /// the source ROI is transformed into by the affine transformation expressed
    /// by the coefficients array (aCoeffs).
    /// </summary>
    /// <param name="aSrcROI">The source ROI.</param>
    /// <param name="aQuad">The resulting destination quadrangle.</param>
    /// <param name="aCoeffs">The afine transform coefficients.</param>
    MPPErrorCode mppciGetAffineQuad(MppiRect aSrcROI, double aQuad[4][2], const double aCoeffs[2][3]);

    /// <summary>
    /// Compute bounding-box of transformed image.
    /// <para/>
    /// The method effectively computes the bounding box (axis aligned rectangle) of
    /// the transformed source ROI (see mppciGetAffineQuad()).
    /// </summary>
    /// <param name="aSrcROI">The source ROI.</param>
    /// <param name="aBoundingBox">The resulting bounding box.</param>
    /// <param name="aCoeffs">The affine transform coefficients.</param>
    MPPErrorCode mppciGetAffineBound(MppiRect aSrcROI, double aBoundingBox[2][2], const double aCoeffs[2][3]);

    /// <summary>
    /// Calculates perspective transform coefficients given source rectangular ROI
    /// and its destination quadrangle projection
    /// </summary>
    /// <param name="aSrcROI">The source ROI.</param>
    /// <param name="aQuad">The destination quadrilateral.</param>
    /// <param name="aCoeffs">Perspective transform coefficients</param>
    MPPErrorCode mppciGetPerspectiveTransform(MppiRect aSrcROI, const double aQuad[4][2], double aCoeffs[3][3]);

    /// <summary>
    /// Calculates perspective transform projection of given source rectangular ROI
    /// </summary>
    /// <param name="aSrcROI">The source ROI.</param>
    /// <param name="aQuad">The resulting destination quadrangle.</param>
    /// <param name="aCoeffs">Perspective transform coefficients</param>
    MPPErrorCode mppciGetPerspectiveQuad(MppiRect aSrcROI, double aQuad[4][2], const double aCoeffs[3][3]);

    /// <summary>
    /// Calculates bounding box of the perspective transform projection of the given source rectangular ROI
    /// </summary>
    /// <param name="aSrcROI">The source ROI.</param>
    /// <param name="aBoundingBox">The resulting bounding box.</param>
    /// <param name="aCoeffs">Perspective transform coefficients</param>
    MPPErrorCode mppciGetPerspectiveBound(MppiRect aSrcROI, double aBoundingBox[2][2], const double aCoeffs[3][3]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for RGBtoXYZ colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToXYZ.<para/>
    /// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoXYZ(float aTwistMatrix[3][3]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for BGRtoXYZ colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToXYZ but adjusted for BGR channel order in source image.<para/>
    /// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoXYZ(float aTwistMatrix[3][3]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for XYZtoRGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's XYZToRGB.<para/>
    /// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixXYZtoRGB(float aTwistMatrix[3][3]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for XYZtoBGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's XYZToRGB but adjusted for BGR channel order in destination image.<para/>
    /// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixXYZtoBGR(float aTwistMatrix[3][3]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for RGB to PhotoYCC colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCC.<para/>
    /// Values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html<para/>
    /// Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for BGR to PhotoYCC colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCC but adjusted for BGR channel order in source image.<para/>
    /// Values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html<para/>
    /// Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for PhotoYCC to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCC.<para/>
    /// Values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html, scaled
    /// by 1.3847 to match the actual output.<para/> Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCCtoRGB_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for PhotoYCC to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCC but adjusted for BGR channel order in destination image.<para/>
    /// Values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html, scaled
    /// by 1.3847 to match the actual output.<para/> Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCCtoBGR_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YUV colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYUV.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for U: [-112..112] shifted by +128.<para/>
    /// The value range for V: [-157..157] shifted by +128 and saturated to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYUV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YUV colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYUV.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for U: [-112..112] shifted by +128.<para/>
    /// The value range for V: [-157..157] shifted by +128 and saturated to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYUV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YUV to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YUVToRGB.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for U: [-112..112] shifted by +128.<para/>
    /// The value range for V: [-157..157] shifted by +128 and saturated to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYUVtoRGB_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YUV to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YUVToBGR.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for U: [-112..112] shifted by +128 to [16..240].<para/>
    /// The value range for V: [-157..157] shifted by +128 and saturated to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYUVtoBGR_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (no specific suffix).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCbCr_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (no specific suffix).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCbCr_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (no specific suffix).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoRGB_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (no specific suffix).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoBGR_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (no specific suffix) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCrCb_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (no specific suffix) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCrCb_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (no specific suffix) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoRGB_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (no specific suffix) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html which itself
    /// refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology Publishing, 3rd
    /// Edition, 2001.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoBGR_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (709CSC suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709csc.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCbCr_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (709CSC suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709csc.html.
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCbCr_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (709CSC suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/ycbcrtobgr-709csc.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoRGB_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (709CSC suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/ycbcrtobgr-709csc.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoBGR_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (709CSC suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709csc.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCrCb_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (709CSC suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709csc.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCrCb_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (709CSC suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/ycbcrtobgr-709csc.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoRGB_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (709CSC suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/ycbcrtobgr-709csc.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoBGR_CSC_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (JPEG suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCbCr_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (JPEG suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCbCr_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (JPEG suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoRGB_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (JPEG suffix in NPP).<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoBGR_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (JPEG suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCrCb_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (JPEG suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCrCb_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (JPEG suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoRGB_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (JPEG suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [0..255]<para/>
    /// The value range for Y: [0..255]<para/>
    /// The value range for CbCr: [-128..127] shifted by +128 to [0..255].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoBGR_JPEG_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (709HDTV suffix in NPP).<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCbCr_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCbCr colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (709HDTV suffix in NPP).<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/cbycr422tobgr-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCbCr_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (709HDTV suffix in NPP).<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/cbycr422tobgr-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoRGB_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCbCr to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (709HDTV suffix in NPP).<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/cbycr422tobgr-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCbCrtoBGR_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for RGB to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's RGBToYCbCr (709HDTV suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixRGBtoYCrCb_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for BGR to YCrCb colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's BGRToYCbCr (709HDTV suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixBGRtoYCrCb_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to RGB colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToRGB (709HDTV suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/cbycr422tobgr-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoRGB_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist3x4 function for YCrCb to BGR colorspace
    /// conversion.<para/>
    /// Gives the same results as NPP's YCbCrToBGR (709HDTV suffix in NPP) but adjusted for CrCb.<para/>
    /// The value range for RGB: [16..235]<para/>
    /// The value range for Y: [16..235]<para/>
    /// The value range for CbCr: [-112..112] shifted by +128 to [16..240].<para/>
    /// Values are specific for 8-bit unsigned integer types.<para/>
    /// The same values as in
    /// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/cbycr422tobgr-709hdtv.html
    /// </summary>
    /// <param name="aTwistMatrix">Output for use in ColorTwist function</param>
    MPPErrorCode mppciGetMatrixYCrCbtoBGR_HDTV_8u(float aTwistMatrix[3][4]);

    /// <summary>
    /// Returns the last error code produced by a MPP API call.<para/>
    /// The error code is specific to the calling CPU thread. If an API call returned a non-zero error code, this
    /// function must be called by the same host thread for inspection.
    /// </summary>
    /// <returns>MPP C-API error code</returns>
    MPPErrorCode mppcGetLastErrorCode();

    /// <summary>
    /// Returns the last error message produced by a MPP API call.<para/>
    /// The error message is specific to the calling CPU thread. If an API call returned a non-zero error code, this
    /// function must be called by the same host thread for inspection.
    /// </summary>
    /// <returns>Detailed error message associated to the lastest error.</returns>
    const char *mppcGetLastErrorMessage();

    /// <summary>
    /// Returns in aStreamCtx a stream context associated with the current CUDA context, the current device and the
    /// currently set default stream.<para/> When the currently active cuda context switches, this stream context is not
    /// valid until the original context is set to current again.<para/> The stream context fields can also be filled
    /// manually with the corresponding CUDA-API calls.
    /// </summary>
    /// <param name="aStreamCtx">MPP stream context.</param>
    /// <returns>MPP C-API error code</returns>
    MPPErrorCode mppcGetStreamContext(MppStreamCtx *aStreamCtx);

    /// <summary>
    /// Sets the default stream to use in case no stream context (nullptr) is passed to a MPP funtion.
    /// </summary>
    /// <param name="aStream">CUDA stream to set as default.</param>
    /// <returns>MPP C-API error code</returns>
    MPPErrorCode mppcSetDefaultStream(cudaStream_t aStream);

    /// <summary>
    /// When the main application activates another CUDA context or sets another CUDA device active, and if the default
    /// context is used, then it is mandatory to update the internally used default stream context to the new values
    /// with this API call.<para/>
    /// The first call to get the default stream context is valid without a prior call to mppcUpdateContext(), though.
    /// </summary>
    /// <returns>MPP C-API error code</returns>
    MPPErrorCode mppcUpdateContext();

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_SUPPORT_H
