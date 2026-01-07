#pragma once
#include <common/defines.h>
#include <common/image/matrix.h>
#include <common/mpp_defs.h>

namespace mpp::image::color
{
// swap channel 1 and 3, e.g. RGB to BGR
DEVICE_ONLY_CODE constexpr Matrix<float> Swap1_3(0, 0, 1, 0, 1, 0, 1, 0, 0);

// swap channel 2 and 3, e.g. YCbCr to YCrCb
DEVICE_ONLY_CODE constexpr Matrix<float> Swap2_3(1, 0, 0, 0, 0, 1, 0, 1, 0);

// Offset 8u YCbCr with limited range: 16 128 128
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetYCbCrLR(16, 128, 128);

// Offset 8u YCbCr with full range: 0 128 128
DEVICE_ONLY_CODE constexpr Vector3<float> OffsetYCbCrFR(0, 128, 128);

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
DEVICE_ONLY_CODE constexpr Matrix<float> RGBtoYUV(0.299f, 0.587f, 0.114f,   //
                                                  -0.147f, -0.289f, 0.436f, //
                                                  0.615f, -0.515f, -0.100f);
DEVICE_ONLY_CODE constexpr Matrix<float> YUVtoRGB(1.0f, 0.0f, 1.140f,     //
                                                  1.0f, -0.394f, -0.581f, //
                                                  1.0f, 2.032f, 0.0f);

// values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
DEVICE_ONLY_CODE constexpr Matrix<float> RGBtoXYZ(0.412453f, 0.35758f, 0.180423f, //
                                                  0.212671f, 0.71516f, 0.072169f, //
                                                  0.019334f, 0.119193f, 0.950227f);
DEVICE_ONLY_CODE constexpr Matrix<float> XYZtoRGB(3.240479f, -1.53715f, -0.498535f, //
                                                  -0.969256f, 1.875991f, 0.041556f, //
                                                  0.055648f, -0.204043f, 1.057311f);

// values as in https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-1/color-models.html
DEVICE_ONLY_CODE constexpr Matrix<float> RGBtoYCC(0.213f, 0.419f, 0.081f,   //
                                                  -0.131f, -0.256f, 0.387f, //
                                                  0.373f, -0.312f, -0.061f);
DEVICE_ONLY_CODE constexpr Matrix<float> YCCtoRGB(0.981f, 0.0f, 1.315f,     //
                                                  0.981f, -0.311f, -0.669f, //
                                                  0.981f, 1.601f, 0.0f);
} // namespace mpp::image::color