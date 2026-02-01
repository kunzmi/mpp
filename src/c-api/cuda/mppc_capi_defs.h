#ifndef MPPDEFS_H
#define MPPDEFS_H

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // NOLINTBEGIN(modernize-use-using,performance-enum-size)

    typedef unsigned char Mpp8u;   // 8-bit unsigned chars
    typedef signed char Mpp8s;     // 8-bit signed chars
    typedef unsigned short Mpp16u; // 16-bit unsigned integers
    typedef short Mpp16s;          // 16-bit signed integers
    typedef unsigned int Mpp32u;   // 32-bit unsigned integers
    typedef int Mpp32s;            // 32-bit signed integers
    typedef uint64_t Mpp64u;       // 64-bit unsigned integers
    typedef int64_t Mpp64s;        // 64-bit signed integers
    typedef __half Mpp16f;         // 16-bit half precision floating-point number
    typedef __nv_bfloat16 Mpp16bf; // 16-bit BFLOAT floating-point number
    typedef float Mpp32f;          // 32-bit (IEEE) floating-point numbers
    typedef double Mpp64f;         // 64-bit floating-point numbers

    typedef unsigned char *DevPtrMpp8u;   // 8-bit unsigned chars
    typedef signed char *DevPtrMpp8s;     // 8-bit signed chars
    typedef unsigned short *DevPtrMpp16u; // 16-bit unsigned integers
    typedef short *DevPtrMpp16s;          // 16-bit signed integers
    typedef unsigned int *DevPtrMpp32u;   // 32-bit unsigned integers
    typedef int *DevPtrMpp32s;            // 32-bit signed integers
    typedef uint64_t *DevPtrMpp64u;       // 64-bit unsigned integers
    typedef int64_t *DevPtrMpp64s;        // 64-bit signed integers
    typedef __half *DevPtrMpp16f;         // 16-bit half precision floating-point number
    typedef __nv_bfloat16 *DevPtrMpp16bf; // 16-bit BFLOAT floating-point number
    typedef float *DevPtrMpp32f;          // 32-bit (IEEE) floating-point numbers
    typedef double *DevPtrMpp64f;         // 64-bit floating-point numbers

    typedef const unsigned char *ConstDevPtrMpp8u;   // 8-bit unsigned chars
    typedef const signed char *ConstDevPtrMpp8s;     // 8-bit signed chars
    typedef const unsigned short *ConstDevPtrMpp16u; // 16-bit unsigned integers
    typedef const short *ConstDevPtrMpp16s;          // 16-bit signed integers
    typedef const unsigned int *ConstDevPtrMpp32u;   // 32-bit unsigned integers
    typedef const int *ConstDevPtrMpp32s;            // 32-bit signed integers
    typedef const uint64_t *ConstDevPtrMpp64u;       // 64-bit unsigned integers
    typedef const int64_t *ConstDevPtrMpp64s;        // 64-bit signed integers
    typedef const __half *ConstDevPtrMpp16f;         // 16-bit half precision floating-point number
    typedef const __nv_bfloat16 *ConstDevPtrMpp16bf; // 16-bit BFLOAT floating-point number
    typedef const float *ConstDevPtrMpp32f;          // 32-bit (IEEE) floating-point numbers
    typedef const double *ConstDevPtrMpp64f;         // 64-bit floating-point numbers

    /// <summary>
    /// Complex Number
    /// This struct represents a short complex number.
    /// </summary>
    typedef struct
    {
        Mpp16s re;
        Mpp16s im;
    } Mpp16sc;
    typedef Mpp16sc *DevPtrMpp16sc;
    typedef const Mpp16sc *ConstDevPtrMpp16sc;

    /// <summary>
    /// Complex Number
    /// This struct represents an int complex number.
    /// </summary>
    typedef struct
    {
        Mpp32s re;
        Mpp32s im;
    } Mpp32sc;
    typedef Mpp32sc *DevPtrMpp32sc;
    typedef const Mpp32sc *ConstDevPtrMpp32sc;

    /// <summary>
    /// Complex Number
    /// This struct represents an long64 complex number.
    /// </summary>
    typedef struct
    {
        Mpp64s re;
        Mpp64s im;
    } Mpp64sc;
    typedef Mpp64sc *DevPtrMpp64sc;
    typedef const Mpp64sc *ConstDevPtrMpp64sc;

    /// <summary>
    /// Complex Number
    /// This struct represents a float complex number.
    /// </summary>
    typedef struct
    {
        Mpp32f re;
        Mpp32f im;
    } Mpp32fc;
    typedef Mpp32fc *DevPtrMpp32fc;
    typedef const Mpp32fc *ConstDevPtrMpp32fc;

    /// <summary>
    /// Complex Number
    /// This struct represents a double complex number.
    /// </summary>
    typedef struct
    {
        Mpp64f re;
        Mpp64f im;
    } Mpp64fc;
    typedef Mpp64fc *DevPtrMpp64fc;
    typedef const Mpp64fc *ConstDevPtrMpp64fc;

    /// <summary>
    /// Rounding Modes<para/>
    /// The enumerated rounding modes are used by a large number of MPP primitives
    /// to allow the user to specify the method by which fractional values are converted
    /// to integer values.
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Round to the nearest even integer.<para/>
        /// All fractional numbers are rounded to their nearest integer. The ambiguous
        /// cases (i.e. integer.5) are rounded to the closest even integer.<para/>
        /// C++ nearbyint() function with FE_TONEAREST rounding mode enabled on x86 CPUs<para/>
        /// E.g.<para/>
        /// - roundNear(0.4) = 0<para/>
        /// - roundNear(0.5) = 0<para/>
        /// - roundNear(0.6) = 1<para/>
        /// - roundNear(1.5) = 2<para/>
        /// - roundNear(1.9) = 2<para/>
        /// - roundNear(-1.5) = -2<para/>
        /// - roundNear(-2.5) = -2<para/>
        /// </summary>
        MPP_ROUND_NearestTiesToEven,
        /// <summary>
        /// Round according to financial rule.<para/>
        /// All fractional numbers are rounded to their nearest integer. The ambiguous
        /// cases (i.e. integer.5) are rounded away from zero.<para/>
        /// C++ round() function<para/>
        /// E.g.<para/>
        /// - roundNearestTiesAwayFromZero(0.4)  = 0<para/>
        /// - roundNearestTiesAwayFromZero(0.5)  = 1<para/>
        /// - roundNearestTiesAwayFromZero(0.6)  = 1<para/>
        /// - roundNearestTiesAwayFromZero(1.5)  = 2<para/>
        /// - roundNearestTiesAwayFromZero(1.9)  = 2<para/>
        /// - roundNearestTiesAwayFromZero(-1.5) = -2<para/>
        /// - roundNearestTiesAwayFromZero(-2.5) = -3<para/>
        /// </summary>
        MPP_ROUND_NearestTiesAwayFromZero,
        /// <summary>
        /// Round towards zero (truncation).<para/>
        /// All fractional numbers of the form integer. Decimals are truncated to
        /// integer.<para/>
        /// C++ trunc() function<para/>
        /// - roundZero(0.4)  = 0<para/>
        /// - roundZero(0.5)  = 0<para/>
        /// - roundZero(0.6)  = 0<para/>
        /// - roundZero(1.5)  = 1<para/>
        /// - roundZero(1.9)  = 1<para/>
        /// - roundZero(-1.5) = -1<para/>
        /// - roundZero(-2.5) = -2<para/>
        /// </summary>
        MPP_ROUND_TowardZero,
        /// <summary>
        /// Round towards negative infinity.<para/>
        /// C++ floor() function<para/>
        /// E.g.<para/>
        /// - floor(0.4)  = 0<para/>
        /// - floor(0.5)  = 0<para/>
        /// - floor(0.6)  = 0<para/>
        /// - floor(1.5)  = 1<para/>
        /// - floor(1.9)  = 1<para/>
        /// - floor(-1.5) = -2<para/>
        /// - floor(-2.5) = -3<para/>
        /// </summary>
        MPP_ROUND_TowardNegativeInfinity,
        /// <summary>
        /// Round towards positive infinity.<para/>
        /// C++ ceil() function<para/>
        /// E.g.<para/>
        /// - ceil(0.4)  = 1<para/>
        /// - ceil(0.5)  = 1<para/>
        /// - ceil(0.6)  = 1<para/>
        /// - ceil(1.5)  = 2<para/>
        /// - ceil(1.9)  = 2<para/>
        /// - ceil(-1.5) = -1<para/>
        /// - ceil(-2.5) = -2<para/>
        /// </summary>
        MPP_ROUND_TowardPositiveInfinity
    } MPPRoundingMode;

    /// <summary>
    /// Different Alpha compositing operations
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// OVER compositing operation.<para/>
        /// A occludes B.<para/>
        /// result pixel = alphaA * A + (1 - alphaA) * alphaB * B<para/>
        /// result alpha = alphaA + (1 - alphaA) * alphaB
        /// </summary>
        MPP_OP_ALPHA_Over,
        /// <summary>
        /// IN compositing operation.<para/>
        /// A within B. A acts as a matte for B. A shows only where B is visible.<para/>
        /// result pixel = alphaA * A * alphaB<para/>
        /// result alpha = alphaA * alphaB
        /// </summary>
        MPP_OP_ALPHA_In,
        /// <summary>
        /// OUT compositing operation.<para/>
        /// A outside B. NOT-B acts as a matte for A. A shows only where B is not visible.<para/>
        /// result pixel = alphaA * A * (1 - alphaB)<para/>
        /// result alpha = alphaA * (1 - alphaB)
        /// </summary>
        MPP_OP_ALPHA_Out,
        /// <summary>
        /// ATOP compositing operation.<para/>
        /// Combination of (A IN B) and (B OUT A). B is both back-ground and matte for A.<para/>
        /// result pixel = alphaA * A * alphaB + (1 - alphaA) * alphaB * B<para/>
        /// result alpha = alphaA * alphaB + (1 - alphaA) * alphaB
        /// </summary>
        MPP_OP_ALPHA_ATop,
        /// <summary>
        /// XOR compositing operation.<para/>
        /// Combination of (A OUT B) and (B OUT A). A and B mutually exclude each other.<para/>
        /// result pixel = alphaA * A * (1 - alphaB) + (1 - alphaA) * alphaB * B<para/>
        /// result alpha = alphaA * (1 - alphaB) + (1 - alphaA) * alphaB
        /// </summary>
        MPP_OP_ALPHA_XOr,
        /// <summary>
        /// PLUS compositing operation.<para/>
        /// Blend without precedence.<para/>
        /// result pixel = alphaA * A + alphaB * B<para/>
        /// result alpha = alphaA + alphaB
        /// </summary>
        MPP_OP_ALPHA_Plus,
        /// <summary>
        /// OVER compositing operation with pre-multiplied pixel values.<para/>
        /// result pixel = A + (1 - alphaA) * B<para/>
        /// result alpha = alphaA + (1 - alphaA) * aB
        /// </summary>
        MPP_OP_ALPHA_OverPremul,
        /// <summary>
        /// IN compositing operation with pre-multiplied pixel values.<para/>
        /// A within B. A acts as a matte for B. A shows only where B is visible.<para/>
        /// result pixel = A * alphaB<para/>
        /// result alpha = alphaA * alphaB
        /// </summary>
        MPP_OP_ALPHA_InPremul,
        /// <summary>
        /// OUT compositing operation with pre-multiplied pixel values.<para/>
        /// A outside B. NOT-B acts as a matte for A. A shows only where B is not visible.<para/>
        /// result pixel = A * (1 - alphaB)<para/>
        /// result alpha = alphaA * (1 - alphaB)
        /// </summary>
        MPP_OP_ALPHA_OutPremul,
        /// <summary>
        /// ATOP compositing operation with pre-multiplied pixel values.<para/>
        /// Combination of (A IN B) and (B OUT A). B is both back-ground and matte for A.<para/>
        /// result pixel = A * alphaB + (1 - alphaA) * B<para/>
        /// result alpha = alphaA * alphaB + (1 - alphaA) * alphaB
        /// </summary>
        MPP_OP_ALPHA_ATopPremul,
        /// <summary>
        /// XOR compositing operation with pre-multiplied pixel values.<para/>
        /// Combination of (A OUT B) and (B OUT A). A and B mutually exclude each other.<para/>
        /// result pixel = A * (1 - alphaB) + (1 - alphaA) * B<para/>
        /// result alpha = alphaA * (1 - alphaB) + (1 - alphaA) * alphaB
        /// </summary>
        MPP_OP_ALPHA_XOrPremul,
        /// <summary>
        /// PLUS compositing operation with pre-multiplied pixel values.<para/>
        /// Blend without precedence.<para/>
        /// result pixel = A + B<para/>
        /// result alpha = alphaA + alphaB
        /// </summary>
        MPP_OP_ALPHA_PlusPremul
    } MPPAlphaOp;

    /// <summary>
    /// Bayer grid position registration
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Blue Green<para/>
        /// Green Red<para/>
        /// Bayer pattern.
        /// </summary>
        MPP_BAYER_BGGR,
        /// <summary>
        /// Red Green<para/>
        /// Green Blue<para/>
        /// Bayer pattern.
        /// </summary>
        MPP_BAYER_RGGB,
        /// <summary>
        /// Green Blue<para/>
        /// Red Green<para/>
        /// Bayer pattern.
        /// </summary>
        MPP_BAYER_GBRG,
        /// <summary>
        /// Green Red<para/>
        /// Blue Green<para/>
        /// Bayer pattern.
        /// </summary>
        MPP_BAYER_GRBG
    } MPPBayerGridPosition;

    /// <summary>
    /// Mirror direction control
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Flip around horizontal axis in mirror function.
        /// </summary>
        MPP_MIRROR_Horizontal,
        /// <summary>
        /// Flip around vertical axis in mirror function.
        /// </summary>
        MPP_MIRROR_Vertical,
        /// <summary>
        /// Flip around both axes in mirror function.
        /// </summary>
        MPP_MIRROR_Both
    } MPPMirrorAxis;

    /// <summary>
    /// Pixel comparison control values
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Returns true if the pixel value is &lt; than the value to compare with.
        /// </summary>
        MPP_COMP_Less = 1u,
        /// <summary>
        /// Returns true if the pixel value is &lt;= than the value to compare with.
        /// </summary>
        MPP_COMP_LessEq = 1u << 1u,
        /// <summary>
        /// Returns true if the pixel value is == than the value to compare with.
        /// </summary>
        MPP_COMP_Eq = 1u << 2u,
        /// <summary>
        /// Returns true if the pixel value is &gt; than the value to compare with.
        /// </summary>
        MPP_COMP_Greater = 1u << 3u,
        /// <summary>
        /// Returns true if the pixel value is &gt;= than the value to compare with.
        /// </summary>
        MPP_COMP_GreaterEq = 1u << 4u,
        /// <summary>
        /// Returns true if the pixel value is != than the value to compare with.
        /// </summary>
        MPP_COMP_NEq = 1u << 5u,
        /// <summary>
        /// Returns true if the pixel value is finite (only for floating point).
        /// </summary>
        MPP_COMP_IsFinite = 1u << 6u,
        /// <summary>
        /// Returns true if the pixel value is NaN (only for floating point).
        /// </summary>
        MPP_COMP_IsNaN = 1u << 7u,
        /// <summary>
        /// Returns true if the pixel value is infinite (only for floating point).
        /// </summary>
        MPP_COMP_IsInf = 1u << 8u,
        /// <summary>
        /// Returns true if the pixel value is infinite or NaN (i.e. not finite) (only for floating point).
        /// </summary>
        MPP_COMP_IsInfOrNaN = 1u << 9u,
        /// <summary>
        /// Returns true if the pixel value is positive infinite (only for floating point).
        /// </summary>
        MPP_COMP_IsPositiveInf = 1u << 10u,
        /// <summary>
        /// Returns true if the pixel value is negative infinite (only for floating point).
        /// </summary>
        MPP_COMP_IsNegativeInf = 1u << 11u,
        /// <summary>
        /// If PerChannel flag is set, the comparison is performed per channel independently.
        /// </summary>
        MPP_COMP_PerChannel = 1u << 26u,
        /// <summary>
        /// If AnyChannel flag is set, the comparison returns true if any of the pixel channel comparisons is true. If
        /// not set, all pixel channel comparisons must be true.
        /// </summary>
        MPP_COMP_AnyChannel = 1u << 27u
    } MPPCompareOp;

    /// <summary>
    /// Border modes for image filtering<para/>
    /// Note: NPP currently only supports NPP_BORDER_REPLICATE, why we will base the enum values on IPP instead:
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Undefined image border type.
        /// </summary>
        MPP_BORDER_None,
        /// <summary>
        /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
        /// c c c | 0 1 2 3 | c c c for a constant c
        /// </summary>
        MPP_BORDER_Constant,
        /// <summary>
        /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
        /// 0 0 0 | 0 1 2 3 | 3 3 3
        /// </summary>
        MPP_BORDER_Replicate,
        /// <summary>
        /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
        /// 3 2 1 | 0 1 2 3 | 2 1 0
        /// </summary>
        MPP_BORDER_Mirror,
        /// <summary>
        /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
        /// 2 1 0 | 0 1 2 3 | 3 2 1
        /// </summary>
        MPP_BORDER_MirrorReplicate,
        /// <summary>
        /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
        /// 1 2 3 | 0 1 2 3 | 0 1 2
        /// </summary>
        MPP_BORDER_Wrap,
        /// <summary>
        /// In SmoothEdge border type, all pixels that fall outside the input image ROI are ignored and the destination
        /// pixel is not written. Except for the one pixel line sourrounding the input ROI, here the image pixels are
        /// extrapolated in order to obtain "a smooth edge". It is not exactly the same algorithm as in IPP, but
        /// similar. Note: In NPP and IPP, SmoothEdge is a flag for interpolation mode, but a member in BorderType seems
        /// more reasonable...
        /// </summary>
        MPP_BORDER_SmoothEdge
    } MPPBorderType;

    /// <summary>
    /// Defines how to evenly spread an integer distribution<para/>
    /// NPP uses a different definition on how two create evenly spaced bins for histograms for integer data. MPP
    /// supports both definitions in its EvenLevels function.
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Tries to reproduce the same even distribution as used in the CUB implementiation (used as backend for
        /// HistogramEven)
        /// </summary>
        MPP_HIST_Default,
        /// <summary>
        /// Tries to reproduce the same even distribution as used in the NPP implementation of EvenLevels and
        /// HistogramEven
        /// </summary>
        MPP_HIST_NPP
    } MPPHistorgamEvenMode;

    /// <summary>
    /// Pixel interpolation modes
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Undefined interpolation mode
        /// </summary>
        MPP_INTER_Undefined = 0,
        /// <summary>
        /// Nearest Neighbor interpolation mode
        /// </summary>
        MPP_INTER_NearestNeighbor = 1,
        /// <summary>
        /// Bi-Linear interpolation mode
        /// </summary>
        MPP_INTER_Linear = 2,
        /// <summary>
        /// Bi-Cubic interpolation mode using Cubic Hermite Splines, named 'cubic' in Matlab
        /// </summary>
        MPP_INTER_CubicHermiteSpline = 3,
        /// <summary>
        /// Bi-Cubic interpolation mode using Lagrange polynomials, named 'Cubic' in NPP (and IPP?)
        /// </summary>
        MPP_INTER_CubicLagrange = 4,
        /// <summary>
        /// Bi-Cubic interpolation mode with two-parameter cubic filter (B=1, C=0)
        /// </summary>
        MPP_INTER_Cubic2ParamBSpline = 5,
        /// <summary>
        /// Bi-Cubic interpolation mode with two-parameter cubic filter (B=0, C=1/2)
        /// </summary>
        MPP_INTER_Cubic2ParamCatmullRom = 6,
        /// <summary>
        /// Bi-Cubic interpolation mode with two-parameter cubic filter (B=1/2, C=3/10)
        /// </summary>
        MPP_INTER_Cubic2ParamB05C03 = 7,
        /// <summary>
        /// Super Sampling interpolation mode<para/>
        /// Note: The super sampling interpolation mode can only be used if width and height of the destination image
        /// are smaller than the source image.
        /// </summary>
        MPP_INTER_Super = 8,
        /// <summary>
        /// Interpolation with the 2-lobed Lanczos Window Function<para/>
        /// The interpolation algorithm uses source image intensities at 16 pixels in the neighborhood of the point in
        /// the source image.
        /// </summary>
        MPP_INTER_Lanczos2Lobed = 9,
        /// <summary>
        /// Interpolation with the 3-lobed Lanczos Window Function<para/>
        /// The interpolation algorithm uses source image intensities at 36 pixels in the neighborhood of the point in
        /// the source image. Note: This is the same as Lanczos in NPP.
        /// </summary>
        MPP_INTER_Lanczos3Lobed = 10
    } MPPInterpolationMode;

    /// <summary>
    /// Mask sizes for fixed size filters
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// 1 x 3 filter mask.
        /// </summary>
        MPP_MASK_SIZE_1x3,
        /// <summary>
        /// 1 x 5 filter mask.
        /// </summary>
        MPP_MASK_SIZE_1x5,
        /// <summary>
        /// 3 x 1 filter mask.
        /// </summary>
        MPP_MASK_SIZE_3x1 = 100,
        /// <summary>
        /// 5 x 1 filter mask.
        /// </summary>
        MPP_MASK_SIZE_5x1,
        /// <summary>
        /// 3 x 3 filter mask.
        /// </summary>
        MPP_MASK_SIZE_3x3 = 200,
        /// <summary>
        /// 5 x 5 filter mask.
        /// </summary>
        MPP_MASK_SIZE_5x5,
        /// <summary>
        /// 7 x 7 filter mask.
        /// </summary>
        MPP_MASK_SIZE_7x7 = 400,
        /// <summary>
        /// 9 x 9 filter mask.
        /// </summary>
        MPP_MASK_SIZE_9x9 = 500,
        /// <summary>
        /// 11 x 11 filter mask.
        /// </summary>
        MPP_MASK_SIZE_11x11 = 600,
        /// <summary>
        /// 13 x 13 filter mask.
        /// </summary>
        MPP_MASK_SIZE_13x13 = 700,
        /// <summary>
        /// 15 x 15 filter mask.
        /// </summary>
        MPP_MASK_SIZE_15x15 = 800
    } MPPMaskSize;

    /// <summary>
    /// Filters with a fixed coeffient matrix
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Gauss filter. Possible mask sizes: 3x3, 5x5, 7x7, 9x9, 11x11, 13x13 and 15x15.<para/>
        /// Filter kernels for Gauss filter are calculated using a sigma value of 0.4 + (filter width / 3.0) *
        /// 0.6.<para/> Note: In NPP the sigma value is given as 0.4 + (filter width / 2) * 0.6, but the values actually
        /// used are "width / 3". Further, the kernel values are normalized to get an exact sum equal to 1.
        /// </summary>
        MPP_FILTER_Gauss,
        /// <summary>
        /// High pass filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        /// -1 -1 -1<para/>
        /// -1 8 -1<para/>
        /// -1 -1 -1<para/>
        /// and<para/>
        /// -1 -1 -1 -1 -1<para/>
        /// -1 -1 -1 -1 -1<para/>
        /// -1 -1 24 -1 -1<para/>
        /// -1 -1 -1 -1 -1<para/>
        /// -1 -1 -1 -1 -1<para/>
        /// </summary>
        MPP_FILTER_HighPass,
        /// <summary>
        /// Low pass filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        /// 1/9 1/9 1/9<para/>
        /// 1/9 1/9 1/9<para/>
        /// 1/9 1/9 1/9<para/>
        /// and<para/>
        /// 1/25 1/25 1/25 1/25 1/25 <para/>
        /// 1/25 1/25 1/25 1/25 1/25 <para/>
        /// 1/25 1/25 1/25 1/25 1/25 <para/>
        /// 1/25 1/25 1/25 1/25 1/25 <para/>
        /// 1/25 1/25 1/25 1/25 1/25 <para/>
        /// </summary>
        MPP_FILTER_LowPass,
        /// <summary>
        /// Laplace filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        /// -1 -1 -1<para/>
        /// -1 8 -1<para/>
        /// -1 -1 -1<para/>
        /// and<para/>
        /// -1 -3 -4 -3 -1<para/>
        /// -3  0  6  0 -3<para/>
        /// -4  6 20  6 -4<para/>
        /// -3  0  6  0 -3<para/>
        /// -1 -3 -4 -3 -1<para/>
        /// </summary>
        MPP_FILTER_Laplace,
        /// <summary>
        /// Horizontal Prewitt filter. Possible mask size: 3x3.<para/>
        /// Used filter is:<para/>
        /// -1 -1 -1<para/>
        ///  0  0  0<para/>
        ///  1  1  1<para/>
        /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation).
        /// MPP uses correlation through out all filtering alike algorithms and in order to obtain the same output as in
        /// NPP, the filter coefficients had to be mirrored.
        /// </summary>
        MPP_FILTER_PrewittHoriz,
        /// <summary>
        /// Vertical Prewitt filter. Possible mask size: 3x3.<para/>
        /// Used filter is:<para/>
        /// -1 0 1<para/>
        /// -1 0 1<para/>
        /// -1 0 1<para/>
        /// Note: The documentation in NPP differs to the actually implemented filter. Nevertheless, this is the filter
        /// used with correlation filtering logic and gives the same results as NPP. MPP uses correlation through out
        /// all filtering alike algorithms and in order to obtain the same output as in NPP, the filter coefficients had
        /// to be mirrored.
        /// </summary>
        MPP_FILTER_PrewittVert,
        /// <summary>
        /// Roberts down filter. Possible mask size: 3x3.<para/>
        /// Used filter is:<para/>
        /// -1 0 0<para/>
        ///  0 1 0<para/>
        ///  0 0 0<para/>
        /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation).
        /// MPP uses correlation through out all filtering alike algorithms and in order to obtain the same output as in
        /// NPP, the filter coefficients had to be mirrored.
        /// </summary>
        MPP_FILTER_RobertsDown,
        /// <summary>
        /// Roberts up filter. Possible mask size: 3x3.<para/>
        /// Used filter is:<para/>
        /// 0 0 -1<para/>
        /// 0 1  0<para/>
        /// 0 0  0<para/>
        /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation).
        /// MPP uses correlation through out all filtering alike algorithms and in order to obtain the same output as in
        /// NPP, the filter coefficients had to be mirrored.
        /// </summary>
        MPP_FILTER_RobertsUp,
        /// <summary>
        /// Horizontal Scharr filter. Possible mask size: 3x3.<para/>
        /// Used filter is:<para/>
        /// -3 -10 -3<para/>
        ///  0   0  0<para/>
        ///  3  10  3<para/>
        /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation).
        /// MPP uses correlation through out all filtering alike algorithms and in order to obtain the same output as in
        /// NPP, the filter coefficients had to be mirrored.
        /// </summary>
        MPP_FILTER_ScharrHoriz,
        /// <summary>
        /// Vertical Prewitt filter. Possible mask size: 3x3.<para/>
        /// Used filter is:<para/>
        ///  -3 0 3<para/>
        /// -10 0 10<para/>
        ///  -3 0 3<para/>
        /// Note: The documentation in NPP differs to the actually implemented filter. Nevertheless, this is the filter
        /// used with correlation filtering logic and gives the same results as NPP. MPP uses correlation through out
        /// all filtering alike algorithms and in order to obtain the same output as in NPP, the filter coefficients had
        /// to be mirrored.
        /// </summary>
        MPP_FILTER_ScharrVert,
        /// <summary>
        /// Sharpen filter. Possible mask size: 3x3.<para/>
        /// Used filter is:<para/>
        /// -1/8 -1/8 -1/8<para/>
        /// -1/8 16/8 -1/8<para/>
        /// -1/8 -1/8 -1/8<para/>
        /// </summary>
        MPP_FILTER_Sharpen,
        /// <summary>
        /// Second cross derivative Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        /// -1 0  1<para/>
        ///  0 0  0<para/>
        ///  1 0 -1<para/>
        /// and<para/>
        /// -1 -2  0  2  1<para/>
        /// -2 -4  0  4  2<para/>
        ///  0  0  0  0  0<para/>
        ///  2  4  0 -4 -2<para/>
        ///  1  2  0 -2 -1<para/>
        /// </summary>
        MPP_FILTER_SobelCross,
        /// <summary>
        /// Horizontal Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        ///  1  2  1<para/>
        ///  0  0  0<para/>
        /// -1 -2 -1<para/>
        /// and<para/>
        /// -1 -4  -6 -4 -1<para/>
        /// -2 -8 -12 -8 -2<para/>
        ///  0  0   0  0  0<para/>
        ///  2  8  12  8  2<para/>
        ///  1  4   6  4  1<para/>
        /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation).
        /// MPP uses correlation through out all filtering alike algorithms and in order to obtain the same output as in
        /// NPP, the filter coefficients had to be mirrored.
        /// </summary>
        MPP_FILTER_SobelHoriz,
        /// <summary>
        /// Vertical Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        /// 1 0 -1<para/>
        /// 2 0 -2<para/>
        /// 1 0 -1<para/>
        /// and<para/>
        /// -1  -2 0  2 1<para/>
        /// -4  -8 0  8 4<para/>
        /// -6 -12 0 12 6<para/>
        /// -4  -8 0  8 4<para/>
        /// -1  -2 0  2 1<para/>
        /// Note: The documentation in NPP differs to the actually implemented filter. Nevertheless, this is the filter
        /// used with correlation filtering logic and gives the same results as NPP. MPP uses correlation through out
        /// all filtering alike algorithms and in order to obtain the same output as in NPP, the filter coefficients had
        /// to be mirrored.
        /// </summary>
        MPP_FILTER_SobelVert,
        /// <summary>
        /// Second derivative horizontal Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        ///  1  2  1<para/>
        /// -2 -4 -2<para/>
        ///  1  2  1<para/>
        /// and<para/>
        ///  1  4   6  4  1<para/>
        ///  0  0   0  0  0<para/>
        /// -2 -8 -12 -8 -2<para/>
        ///  0  0   0  0  0<para/>
        ///  1  4   6  4  1<para/>
        /// </summary>
        MPP_FILTER_SobelHorizSecond,
        /// <summary>
        /// Second derivative vertical Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
        /// Used filters are:<para/>
        /// 1 -2 1<para/>
        /// 2  4 2<para/>
        /// 1 -2 1<para/>
        /// and<para/>
        /// 1  0  -2  0  1<para/>
        /// 4  0  -8  0  4<para/>
        /// 6  0 -12  0  6<para/>
        /// 4  0  -8  0  4<para/>
        /// 1  0  -2  0  1<para/>
        /// </summary>
        MPP_FILTER_SobelVertSecond
    } MPPFixedFilter;

    /// <summary>
    /// Distance norm
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Infinity norm (maximum)
        /// </summary>
        MPP_NORM_Inf,
        /// <summary>
        /// L1 norm (sum of absolute values)
        /// </summary>
        MPP_NORM_L1,
        /// <summary>
        /// L2 norm (square root of sum of squares)
        /// </summary>
        MPP_NORM_L2
    } MPPNorm;

    /// <summary>
    /// MPP Error Codes
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Error free operation
        /// </summary>
        MPP_SUCCESS = 0,
        /// <summary>
        /// Successful operation (same as MPP_SUCCESS)
        /// </summary>
        MPP_NO_ERROR = MPP_SUCCESS,
        /// <summary>
        /// An argument passed to MPP is outside the valid value range.
        /// </summary>
        MPP_ERROR_INVALID_ARGUMENT = -1,
        /// <summary>
        /// MPP received a nullptr but expected a valid pointer.
        /// </summary>
        MPP_ERROR_NULLPTR = -2,
        /// <summary>
        /// An error related to the scratch buffer that was passed to MPP.
        /// </summary>
        MPP_ERROR_SCRATCH_BUFFER = -3,
        /// <summary>
        /// A ROI mismatch for one of the images passed to MPP
        /// </summary>
        MPP_ERROR_ROI = -4,
        /// <summary>
        /// A channel index is out of valid range.
        /// </summary>
        MPP_ERROR_CHANNEL = -5,
        /// <summary>
        /// A FilterArea argument passed to MPP is not valid.
        /// </summary>
        MPP_ERROR_FILTER_AREA = -6,
        /// <summary>
        /// An operation with the coefficients for an affine transformation failed.
        /// </summary>
        MPP_ERROR_AFFINE_TRANSFORMATION = -7,
        /// <summary>
        /// An 3x3-matrix in general or a perspective transformation operation with the provided coefficients failed.
        /// </summary>
        MPP_ERROR_MATRIX = -8,
        /// <summary>
        /// A CUDA API function did not return error code cudaSuccess.
        /// </summary>
        MPP_ERROR_CUDA = -1000,
        /// <summary>
        /// MPP tried to call an unsupported CUDA feature/function.
        /// </summary>
        MPP_ERROR_CUDA_UNSUPPORTED = -1001,
        /// <summary>
        /// An unknown error occured.
        /// </summary>
        MPP_ERROR_UNKNOWN = -999999
    } MPPErrorCode;

    /// <summary>
    /// Chroma sub-sample position: Defines the location of the sub-sampled chroma sample location relative to the fully
    /// sampled Y channel: <para/> Similar as defined in FFMPEG:<para/> Illustration showing the location of the
    /// first(top left) chroma sample of the image, the left shows only luma, the right shows the location of the chroma
    /// sample, the 2 could be imagined to overlay each other but are drawn separately due to limitations of ASCII
    /// <para/>
    /// <para/>
    ///                 1st 2nd       1st 2nd horizontal luma sample positions<para/>
    ///                  v   v         v   v<para/>
    ///                  ______        ______<para/>
    /// 1st luma line > |X   X ...    |3 0 X ...     X are luma samples,<para/>
    ///                 |             |1 2           1-6 are possible chroma positions<para/>
    /// 2nd luma line > |X   X ...    |0 0 X ...     0 is undefined/unknown position
    /// </summary>
    typedef enum
    {
        /// <summary>
        /// Undefined
        /// </summary>
        MPP_CHROMA_Undefined,
        /// <summary>
        /// X: same as luma sample / Y: in-between two sample points
        /// </summary>
        MPP_CHROMA_Left,
        /// <summary>
        /// X: in-between two sample points / Y: in-between two sample points
        /// </summary>
        MPP_CHROMA_Center,
        /// <summary>
        /// X: same as luma sample / Y: same as luma sample
        /// </summary>
        MPP_CHROMA_TopLeft
    } MPPChromaSubsamplePos;

    /// <summary>
    /// Size of an image or Region of interest in an image
    /// </summary>
    typedef struct
    {
        int width;
        int height;
    } MppiSize;

    /// <summary>
    /// Region of interest (ROI) or 2D rectangle defined by a start position x/y and a size (width/height)
    /// </summary>
    typedef struct
    {
        int x;
        int y;
        int width;
        int height;
    } MppiRect;

    /// <summary>
    /// To reduce the number of passed pointers to the Min/Max with Index functions, we define a small structure
    /// containing the computed indices
    /// </summary>
    typedef struct
    {
        int indexMinX;
        int indexMinY;
        int indexMaxX;
        int indexMaxY;
    } MppiIndexMinMax;
    typedef MppiIndexMinMax *DevPtrMppiIndexMinMax;

    /// <summary>
    /// To reduce the number of passed pointers to the Min/Max with Index functions, we define a small structure
    /// containing the computed indices
    /// </summary>
    typedef struct
    {
        int indexMinX;
        int indexMinY;
        int indexMaxX;
        int indexMaxY;
        int channelMin;
        int channelMax;
    } MppiIndexMinMaxChannel;
    typedef MppiIndexMinMaxChannel *DevPtrMppiIndexMinMaxChannel;

    /// <summary>
    /// Combines filter size and center point in one struct for a simplified API
    /// </summary>
    typedef struct
    {
        int width;
        int height;
        int centerX;
        int centerY;
    } MppiFilterArea;

    /// <summary>
    /// Stream context
    /// </summary>
    typedef struct
    {
        /// <summary>
        /// The cuda stream to use for kernel execution
        /// </summary>
        cudaStream_t Stream;

        /// <summary>
        /// From cudaGetDevice()
        /// </summary>
        int DeviceId;

        /// <summary>
        /// From cudaGetDeviceProperties()
        /// </summary>
        int MultiProcessorCount;

        /// <summary>
        /// From cudaGetDeviceProperties()
        /// </summary>
        int MaxThreadsPerMultiProcessor;

        /// <summary>
        /// From cudaGetDeviceProperties()
        /// </summary>
        int MaxThreadsPerBlock;

        /// <summary>
        /// From cudaGetDeviceProperties()
        /// </summary>
        size_t SharedMemPerBlock;

        /// <summary>
        /// From cudaGetDeviceAttribute()
        /// </summary>
        int ComputeCapabilityMajor;

        /// <summary>
        /// From cudaGetDeviceAttribute()
        /// </summary>
        int ComputeCapabilityMinor;

        /// <summary>
        /// From cudaStreamGetFlags()
        /// </summary>
        unsigned int StreamFlags;

        /// <summary>
        /// WarpSize
        /// </summary>
        int WarpSize;
    } MppCudaStreamCtx;
    typedef const MppCudaStreamCtx *CPtrMppCudaStreamCtx;

    // NOLINTEND(modernize-use-using,performance-enum-size)
#ifdef __cplusplus
}
#endif
#endif // MPPDEFS_H