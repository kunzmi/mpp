#pragma once
#include <common/defines.h>
#include <common/image/matrix.h>
#include <common/image/matrix3x4.h>
#include <common/mpp_defs.h>

namespace mpp::image::color
{
#pragma region Helper matrices used in the final defintions:
// swap channel 1 and 3, e.g. RGB to BGR
DEVICE_ONLY_CODE constexpr Matrix<float> Swap1_3(0, 0, 1, 0, 1, 0, 1, 0, 0);

// swap channel 2 and 3, e.g. YCbCr to YCrCb
DEVICE_ONLY_CODE constexpr Matrix<float> Swap2_3(1, 0, 0, 0, 0, 1, 0, 1, 0);

// Offset 8u YCbCr with limited range: 16 128 128
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetYCbCrLR(16, 128, 128);

// Offset 8u YCbCr with full range: 0 128 128
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetYCbCrFR(0, 128, 128);

// Offset 8u RGB with limited range: 16 16 16
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetRGBLR(16, 16, 16);

// Offset 8u YCbCr with limited range: 16 128 128
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetYCbCrLR_Inv(-16, -128, -128);

// Offset 8u YCbCr with full range: 0 128 128
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetYCbCrFR_Inv(0, -128, -128);

// Offset 8u RGB with limited range: 16 16 16
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetRGBLR_Inv(-16, -16, -16);

// values as in
// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709hdtv.html
// 709HDTV in NPP
// LR = LimitedRange
// RGB[16..235] -> Y[16..235]CbCr[16..240]
DEVICE_ONLY_CODE constexpr Matrix<float> RGB_LRtoYCbCrBT709_LR(0.213f, 0.715f, 0.072f,   //
                                                               -0.117f, -0.394f, 0.511f, //
                                                               0.511f, -0.464f, -0.047f);
// values as in
// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/cbycr422tobgr-709hdtv.html
// 709HDTV in NPP
// LR = LimitedRange
// Y[16..235]CbCr[16..240] -> RGB[16..235]
DEVICE_ONLY_CODE constexpr Matrix<float> YCbCrBT709_LRtoRGB_LR(1.0f, 0.0f, 1.540f,     //
                                                               1.0f, -0.183f, -0.459f, //
                                                               1.0f, 1.816f, 0.0f);

// values as in
// https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/bgrtoycbcr420-709csc.html
// 709CSC in NPP
// FR = FullRange / LR = LimitedRange
// RGB[0..255] -> Y[16..235]CbCr[16..240]
DEVICE_ONLY_CODE constexpr Matrix<float> RGB_FRtoYCbCrBT709_LR(0.183f, 0.614f, 0.062f,   //
                                                               -0.101f, -0.338f, 0.439f, //
                                                               0.439f, -0.399f, -0.040f);
// values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/ycbcrtobgr-709csc.html
// 709CSC in NPP
// FR = FullRange / LR = LimitedRange
// Y[16..235]CbCr[16..240] -> RGB[0..255]
DEVICE_ONLY_CODE constexpr Matrix<float> YCbCrBT709_LRtoRGB_FR(1.164f, 0.0f, 1.793f,     //
                                                               1.164f, -0.213f, -0.534f, //
                                                               1.164f, 2.115f, 0.0f);

// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/color-models.html
// which itself refers to Jack, Keith. Video Demystified: a Handbook for the Digital Engineer, LLH Technology
// Publishing, 3rd Edition, 2001.
// No suffix in NPP
// FR = FullRange / LR = LimitedRange
// RGB[0..255] -> Y[16..235]CbCr[16..240]
DEVICE_ONLY_CODE constexpr Matrix<float> RGB_FRtoYCbCrBT601_LR(0.257f, 0.504f, 0.098f,   //
                                                               -0.148f, -0.291f, 0.439f, //
                                                               0.439f, -0.368f, -0.071f);
// No suffix in NPP
// FR = FullRange / LR = LimitedRange
// Y[16..235]CbCr[16..240] -> RGB[0..255]
DEVICE_ONLY_CODE constexpr Matrix<float> YCbCrBT601_LRtoRGB_FR(1.164f, 0.0f, 1.596f,     //
                                                               1.164f, -0.392f, -0.813f, //
                                                               1.164f, 2.017f, 0.0f);

// Values as in https://en.wikipedia.org/wiki/YCbCr which refers to ITU-T T.871
// JPEG in NPP
// FR = FullRange
// RGB[0..255] -> YCbCr[0..255]
DEVICE_ONLY_CODE constexpr Matrix<float> RGB_FRtoYCbCrBT601_FR(0.299f, 0.587f, 0.114f,       //
                                                               -0.168736f, -0.331264f, 0.5f, //
                                                               0.5f, -0.418688f, -0.081312f);
// JPEG in NPP
// FR = FullRange
// YCbCr[0..255] -> RGB[0..255]
DEVICE_ONLY_CODE constexpr Matrix<float> YCbCrBT601_FRtoRGB_FR(1.0f, 0.0f, 1.402f,           //
                                                               1.0f, -0.344136f, -0.714136f, //
                                                               1.0f, 1.772f, 0.0f);

// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
DEVICE_ONLY_CODE constexpr Matrix<float> RGBtoYUVCoeffs(0.299f, 0.587f, 0.114f,   //
                                                        -0.147f, -0.289f, 0.436f, //
                                                        0.615f, -0.515f, -0.100f);
DEVICE_ONLY_CODE constexpr Matrix<float> YUVtoRGBCoeffs(1.0f, 0.0f, 1.140f,     //
                                                        1.0f, -0.394f, -0.581f, //
                                                        1.0f, 2.032f, 0.0f);

#pragma endregion

#pragma region Matrices to be used in ColorTwist functions:
#pragma region RGB to XYZ
/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for RGBtoXYZ colorspace
/// conversion.<para/>
/// Gives the same results as NPP's RGBToXYZ.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> RGBtoXYZ(0.412453f, 0.35758f, 0.180423f, //
                                                  0.212671f, 0.71516f, 0.072169f, //
                                                  0.019334f, 0.119193f, 0.950227f);

/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for XYZtoRGB colorspace
/// conversion.<para/>
/// Gives the same results as NPP's XYZToRGB.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> XYZtoRGB(3.240479f, -1.53715f, -0.498535f, //
                                                  -0.969256f, 1.875991f, 0.041556f, //
                                                  0.055648f, -0.204043f, 1.057311f);

/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for BGRtoXYZ colorspace
/// conversion.<para/>
/// Gives the same results as NPP's RGBToXYZ but adjusted for BGR channel order in source image.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> BGRtoXYZ = RGBtoXYZ * Swap1_3;

/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for XYZtoRGB colorspace
/// conversion.<para/>
/// Gives the same results as NPP's XYZToRGB.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> XYZtoBGR = Swap1_3 * XYZtoRGB;
#pragma endregion
#pragma region RGB to YCC

/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for RGB to PhotoYCC colorspace
/// conversion.<para/>
/// Gives the same results as NPP's RGBToYCC.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> RGBtoYCC(0.213f, 0.419f, 0.081f,   //
                                                  -0.131f, -0.256f, 0.387f, //
                                                  0.373f, -0.312f, -0.061f);

/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for PhotoYCC to RGB colorspace
/// conversion.<para/>
/// Gives the same results as NPP's RGBToYCC.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> YCCtoRGB(0.981f, 0.0f, 1.315f,     //
                                                  0.981f, -0.311f, -0.669f, //
                                                  0.981f, 1.601f, 0.0f);

/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for BGR to PhotoYCC colorspace
/// conversion.<para/>
/// Gives the same results as NPP's RGBToYCC but adjusted for BGR channel order in source image.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> BGRtoYCC = RGBtoYCC * Swap1_3;

/// <summary>
/// Returns in aTwistMatrix the matrix coefficients to be used in ColorTwist function for PhotoYCC to RGB colorspace
/// conversion.<para/>
/// Gives the same results as NPP's RGBToYCC but adjusted for BGR channel order in destination image.<para/>
/// Values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
/// </summary>
DEVICE_ONLY_CODE constexpr Matrix<float> YCCtoBGR = Swap1_3 * YCCtoRGB;
#pragma endregion
#pragma region RGB to YUV

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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYUV(RGBtoYUVCoeffs, OffsetYCbCrFR);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 YUVtoRGB(YUVtoRGBCoeffs, YUVtoRGBCoeffs *OffsetYCbCrFR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYUV(RGBtoYUVCoeffs *Swap1_3, OffsetYCbCrFR);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 YUVtoBGR(Swap1_3 *YUVtoRGBCoeffs, Swap1_3 *YUVtoRGBCoeffs *OffsetYCbCrFR_Inv);
#pragma endregion
#pragma region RGB to YCbCr
#pragma region no suffix
// No suffix in NPP
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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCbCr(RGB_FRtoYCbCrBT601_LR, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoRGB(YCbCrBT601_LRtoRGB_FR, YCbCrBT601_LRtoRGB_FR *OffsetYCbCrLR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCbCr(RGB_FRtoYCbCrBT601_LR *Swap1_3, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoBGR(Swap1_3 *YCbCrBT601_LRtoRGB_FR,
                                                Swap1_3 *YCbCrBT601_LRtoRGB_FR *OffsetYCbCrLR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCrCb(Swap2_3 *RGB_FRtoYCbCrBT601_LR, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoRGB(YCbCrBT601_LRtoRGB_FR *Swap2_3,
                                                YCbCrBT601_LRtoRGB_FR *Swap2_3 *OffsetYCbCrLR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCrCb(Swap2_3 *RGB_FRtoYCbCrBT601_LR *Swap1_3, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoBGR(Swap1_3 *YCbCrBT601_LRtoRGB_FR *Swap2_3,
                                                Swap1_3 *YCbCrBT601_LRtoRGB_FR *Swap2_3 *OffsetYCbCrLR_Inv);
#pragma endregion
#pragma region CSC
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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCbCr_CSC(RGB_FRtoYCbCrBT709_LR, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoRGB_CSC(YCbCrBT709_LRtoRGB_FR, YCbCrBT709_LRtoRGB_FR *OffsetYCbCrLR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCbCr_CSC(RGB_FRtoYCbCrBT709_LR *Swap1_3, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoBGR_CSC(Swap1_3 *YCbCrBT709_LRtoRGB_FR,
                                                    Swap1_3 *YCbCrBT709_LRtoRGB_FR *OffsetYCbCrLR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCrCb_CSC(Swap2_3 *RGB_FRtoYCbCrBT709_LR, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoRGB_CSC(YCbCrBT709_LRtoRGB_FR *Swap2_3,
                                                    YCbCrBT709_LRtoRGB_FR *Swap2_3 *OffsetYCbCrLR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCrCb_CSC(Swap2_3 *RGB_FRtoYCbCrBT709_LR *Swap1_3, OffsetYCbCrLR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoBGR_CSC(Swap1_3 *YCbCrBT709_LRtoRGB_FR *Swap2_3,
                                                    Swap1_3 *YCbCrBT709_LRtoRGB_FR *Swap2_3 *OffsetYCbCrLR_Inv);
#pragma endregion
#pragma region JPEG
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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCbCr_JPEG(RGB_FRtoYCbCrBT601_FR, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoRGB_JPEG(YCbCrBT601_FRtoRGB_FR, YCbCrBT601_FRtoRGB_FR *OffsetYCbCrFR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCbCr_JPEG(RGB_FRtoYCbCrBT601_FR *Swap1_3, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoBGR_JPEG(Swap1_3 *YCbCrBT601_FRtoRGB_FR,
                                                     Swap1_3 *YCbCrBT601_FRtoRGB_FR *OffsetYCbCrFR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCrCb_JPEG(Swap2_3 *RGB_FRtoYCbCrBT601_FR, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoRGB_JPEG(YCbCrBT601_FRtoRGB_FR *Swap2_3,
                                                     YCbCrBT601_FRtoRGB_FR *Swap2_3 *OffsetYCbCrFR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCrCb_JPEG(Swap2_3 *RGB_FRtoYCbCrBT601_FR *Swap1_3, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoBGR_JPEG(Swap1_3 *YCbCrBT601_FRtoRGB_FR *Swap2_3,
                                                     Swap1_3 *YCbCrBT601_FRtoRGB_FR *Swap2_3 *OffsetYCbCrFR_Inv);
#pragma endregion
#pragma region HDTV
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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCbCr_HDTV(RGB_LRtoYCbCrBT709_LR, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoRGB_HDTV(YCbCrBT709_LRtoRGB_LR, YCbCrBT709_LRtoRGB_LR *OffsetYCbCrFR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCbCr_HDTV(RGB_LRtoYCbCrBT709_LR *Swap1_3, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCbCrtoBGR_HDTV(Swap1_3 *YCbCrBT709_LRtoRGB_LR,
                                                     Swap1_3 *YCbCrBT709_LRtoRGB_LR *OffsetYCbCrFR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 RGBtoYCrCb_HDTV(Swap2_3 *RGB_LRtoYCbCrBT709_LR, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoRGB_HDTV(YCbCrBT709_LRtoRGB_LR *Swap2_3,
                                                     YCbCrBT709_LRtoRGB_LR *Swap2_3 *OffsetYCbCrFR_Inv);

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
DEVICE_ONLY_CODE constexpr Matrix3x4 BGRtoYCrCb_HDTV(Swap2_3 *RGB_LRtoYCbCrBT709_LR *Swap1_3, OffsetYCbCrFR);
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
DEVICE_ONLY_CODE constexpr Matrix3x4 YCrCbtoBGR_HDTV(Swap1_3 *YCbCrBT709_LRtoRGB_LR *Swap2_3,
                                                     Swap1_3 *YCbCrBT709_LRtoRGB_LR *Swap2_3 *OffsetYCbCrFR_Inv);
#pragma endregion
#pragma endregion
#pragma endregion
} // namespace mpp::image::color