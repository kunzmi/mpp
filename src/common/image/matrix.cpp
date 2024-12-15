#include "matrix.h"

namespace opp::image
{
const Matrix<float> RGBtoYUV =
    Matrix<float>(0.299f, 0.587f, 0.114f, -0.147f, -0.289f, 0.436f, 0.615f, -0.515f, -0.100f);
const Matrix<float> YUVtoRGB = Matrix<float>(1.0f, 0.0f, 1.140f, 1.0f, -0.394f, -0.581f, 1.0f, 2.032f, 0.0f);

const Matrix<float> RGBtoYCbCr =
    Matrix<float>(0.257f, 0.504f, 0.098f, -0.148f, -0.291f, +0.439f, 0.439f, -0.368f, -0.071f);
const Matrix<float> CbCrtoRGB = Matrix<float>(1.164f, 0.0f, 1.596f, 1.164f, -0.392f, -0.813f, 1.164f, 2.017f, 0.0f);

const Matrix<float> RGBtoXYZ =
    Matrix<float>(0.412453f, 0.35758f, 0.180423f, 0.212671f, 0.71516f, 0.072169f, 0.019334f, 0.119193f, 0.950227f);
const Matrix<float> XYZtoRGB =
    Matrix<float>(3.240479f, -1.53715f, -0.498535f, -0.969256f, 1.875991f, 0.041556f, 0.055648f, -0.204043f, 1.057311f);
} // namespace opp::image