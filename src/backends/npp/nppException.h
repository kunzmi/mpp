#pragma once
#include <common/exception.h>
#include <filesystem>
#include <nppdefs.h>
#include <sstream>

constexpr bool NPP_TRHOW_ON_WARNING = true;

namespace opp::npp
{
/// <summary>
/// NppException is thrown when a NPP API call via nppSafeCall does not return NPP_SUCCESS.
/// </summary>
class NppException : public OPPException
{
  private:
    static constexpr const char *ConvertErrorCodeToMessage(NppStatus aErrorCode);
    static constexpr const char *ConvertErrorCodeToName(NppStatus aErrorCode);

  public:
    NppException(NppStatus aNppStatus, const std::filesystem::path &aCodeFileName, int aLineNumber,
                 const std::string &aFunctionName);
    NppException(NppStatus aNppStatus, const std::string &aMessage, const std::filesystem::path &aCodeFileName,
                 int aLineNumber, const std::string &aFunctionName);
    NppException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                 const std::string &aFunctionName);
    ~NppException() noexcept override = default;

    NppException(NppException &&)                 = default;
    NppException(const NppException &)            = default;
    NppException &operator=(const NppException &) = delete;
    NppException &operator=(NppException &&)      = delete;
};

constexpr const char *NppException::ConvertErrorCodeToMessage(NppStatus aErrorCode)
{
    switch (aErrorCode)
    {
        case NPP_NOT_SUPPORTED_MODE_ERROR:
        case NPP_INVALID_HOST_POINTER_ERROR:
        case NPP_INVALID_DEVICE_POINTER_ERROR:
        case NPP_LUT_PALETTE_BITSIZE_ERROR:
        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
        case NPP_TEXTURE_BIND_ERROR:
        case NPP_WRONG_INTERSECTION_ROI_ERROR:
        case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
        case NPP_MEMFREE_ERROR:
        case NPP_MEMSET_ERROR:
        case NPP_MEMCPY_ERROR:
        case NPP_ALIGNMENT_ERROR:
        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
        case NPP_RESIZE_FACTOR_ERROR:
        case NPP_INTERPOLATION_ERROR:
        case NPP_MIRROR_FLIP_ERROR:
        case NPP_MOMENT_00_ZERO_ERROR:
        case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
        case NPP_THRESHOLD_ERROR:
        case NPP_CONTEXT_MATCH_ERROR:
        case NPP_FFT_FLAG_ERROR:
        case NPP_FFT_ORDER_ERROR:
        case NPP_SCALE_RANGE_ERROR:
        case NPP_DATA_TYPE_ERROR:
        case NPP_OUT_OFF_RANGE_ERROR:
        case NPP_DIVIDE_BY_ZERO_ERROR:
        case NPP_MEMORY_ALLOCATION_ERR:
        case NPP_NULL_POINTER_ERROR:
        case NPP_RANGE_ERROR:
        case NPP_SIZE_ERROR:
        case NPP_BAD_ARGUMENT_ERROR:
        case NPP_NO_MEMORY_ERROR:
        case NPP_NOT_IMPLEMENTED_ERROR:
        case NPP_ERROR:
        case NPP_ERROR_RESERVED:
            return "";

        case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
            return "ZeroCrossing mode not supported";
        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
            return "Unsupported round mode";
        case NPP_QUALITY_INDEX_ERROR:
            return "Image pixels are constant for quality index";
        case NPP_RESIZE_NO_OPERATION_ERROR:
            return "One of the output image dimensions is less than 1 pixel";
        case NPP_OVERFLOW_ERROR:
            return "Number overflows the upper or lower limit of the data type";
        case NPP_NOT_EVEN_STEP_ERROR:
            return "Step value is not pixel multiple";
        case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
            return "Number of levels for histogram is less than 2";
        case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
            return "Number of levels for LUT is less than 2";
        case NPP_CORRUPTED_DATA_ERROR:
            return "Processed data is corrupted";
        case NPP_CHANNEL_ORDER_ERROR:
            return "Wrong order of the destination channels";
        case NPP_ZERO_MASK_VALUE_ERROR:
            return "All values of the mask are zero";
        case NPP_QUADRANGLE_ERROR:
            return "The quadrangle is nonconvex or degenerates into triangle, line or point";
        case NPP_RECTANGLE_ERROR:
            return "Size of the rectangle region is less than or equal to 1";
        case NPP_COEFFICIENT_ERROR:
            return "Unallowable values of the transformation coefficients";
        case NPP_NUMBER_OF_CHANNELS_ERROR:
            return "Bad or unsupported number of channels";
        case NPP_COI_ERROR:
            return "Channel of interest is not 1, 2, or 3";
        case NPP_DIVISOR_ERROR:
            return "Divisor is equal to zero";
        case NPP_CHANNEL_ERROR:
            return "Illegal channel index";
        case NPP_STRIDE_ERROR:
            return "Stride is less than the row length";
        case NPP_ANCHOR_ERROR:
            return "Anchor point is outside mask";
        case NPP_MASK_SIZE_ERROR:
            return "Lower bound is larger than upper bound";
        case NPP_STEP_ERROR:
            return "Step is less or equal zero";

        /* success */
        case NPP_SUCCESS:
            return "Error free operation";

        /* positive return-codes indicate warnings */
        case NPP_NO_OPERATION_WARNING:
            return "Indicates that no operation was performed";
        case NPP_DIVIDE_BY_ZERO_WARNING:
            return "Divisor is zero however does not terminate the execution";
        case NPP_AFFINE_QUAD_INCORRECT_WARNING:
            return "Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary "
                   "properties. First 3 vertices are used, the fourth vertex discarded.";
        case NPP_WRONG_INTERSECTION_ROI_WARNING:
            return "The given ROI has no interestion with either the source or destination ROI. Thus no operation was "
                   "performed.";
        case NPP_WRONG_INTERSECTION_QUAD_WARNING:
            return "The given quadrangle has no intersection with either the source or destination ROI. Thus no "
                   "operation "
                   "was performed.";
        case NPP_DOUBLE_SIZE_WARNING:
            return "Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI "
                   "width/height "
                   "was modified for proper processing.";
        case NPP_MISALIGNED_DST_ROI_WARNING:
            return "Speed reduction due to uncoalesced memory accesses warning.";
        default:
            return "";
    }
}

constexpr const char *NppException::ConvertErrorCodeToName(NppStatus aErrorCode)
{
    switch (aErrorCode)
    {
        case NPP_NOT_SUPPORTED_MODE_ERROR:
            return "NPP_NOT_SUPPORTED_MODE_ERROR";
        case NPP_INVALID_HOST_POINTER_ERROR:
            return "NPP_INVALID_HOST_POINTER_ERROR";
        case NPP_INVALID_DEVICE_POINTER_ERROR:
            return "NPP_INVALID_DEVICE_POINTER_ERROR";
        case NPP_LUT_PALETTE_BITSIZE_ERROR:
            return "NPP_LUT_PALETTE_BITSIZE_ERROR";
        case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
            return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
            return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
        case NPP_TEXTURE_BIND_ERROR:
            return "NPP_TEXTURE_BIND_ERROR";
        case NPP_WRONG_INTERSECTION_ROI_ERROR:
            return "NPP_WRONG_INTERSECTION_ROI_ERROR";
        case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
            return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
        case NPP_MEMFREE_ERROR:
            return "NPP_MEMFREE_ERROR";
        case NPP_MEMSET_ERROR:
            return "NPP_MEMSET_ERROR";
        case NPP_MEMCPY_ERROR:
            return "NPP_MEMCPY_ERROR";
        case NPP_ALIGNMENT_ERROR:
            return "NPP_ALIGNMENT_ERROR";
        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
            return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
            return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
        case NPP_QUALITY_INDEX_ERROR:
            return "NPP_QUALITY_INDEX_ERROR";
        case NPP_RESIZE_NO_OPERATION_ERROR:
            return "NPP_RESIZE_NO_OPERATION_ERROR";
        case NPP_OVERFLOW_ERROR:
            return "NPP_OVERFLOW_ERROR";
        case NPP_NOT_EVEN_STEP_ERROR:
            return "NPP_NOT_EVEN_STEP_ERROR";
        case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
            return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
        case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
            return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
        case NPP_CORRUPTED_DATA_ERROR:
            return "NPP_CORRUPTED_DATA_ERROR";
        case NPP_CHANNEL_ORDER_ERROR:
            return "NPP_CHANNEL_ORDER_ERROR";
        case NPP_ZERO_MASK_VALUE_ERROR:
            return "NPP_ZERO_MASK_VALUE_ERROR";
        case NPP_QUADRANGLE_ERROR:
            return "NPP_QUADRANGLE_ERROR";
        case NPP_RECTANGLE_ERROR:
            return "NPP_RECTANGLE_ERROR";
        case NPP_COEFFICIENT_ERROR:
            return "NPP_COEFFICIENT_ERROR";
        case NPP_NUMBER_OF_CHANNELS_ERROR:
            return "NPP_NUMBER_OF_CHANNELS_ERROR";
        case NPP_COI_ERROR:
            return "NPP_COI_ERROR";
        case NPP_DIVISOR_ERROR:
            return "NPP_DIVISOR_ERROR";
        case NPP_CHANNEL_ERROR:
            return "NPP_CHANNEL_ERROR";
        case NPP_STRIDE_ERROR:
            return "NPP_STRIDE_ERROR";

        case NPP_ANCHOR_ERROR:
            return "NPP_ANCHOR_ERROR";
        case NPP_MASK_SIZE_ERROR:
            return "NPP_MASK_SIZE_ERROR";

        case NPP_RESIZE_FACTOR_ERROR:
            return "NPP_RESIZE_FACTOR_ERROR";
        case NPP_INTERPOLATION_ERROR:
            return "NPP_INTERPOLATION_ERROR";
        case NPP_MIRROR_FLIP_ERROR:
            return "NPP_MIRROR_FLIP_ERROR";
        case NPP_MOMENT_00_ZERO_ERROR:
            return "NPP_MOMENT_00_ZERO_ERROR";
        case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
            return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
        case NPP_THRESHOLD_ERROR:
            return "NPP_THRESHOLD_ERROR";
        case NPP_CONTEXT_MATCH_ERROR:
            return "NPP_CONTEXT_MATCH_ERROR";
        case NPP_FFT_FLAG_ERROR:
            return "NPP_FFT_FLAG_ERROR";
        case NPP_FFT_ORDER_ERROR:
            return "NPP_FFT_ORDER_ERROR";
        case NPP_STEP_ERROR:
            return "NPP_STEP_ERROR";
        case NPP_SCALE_RANGE_ERROR:
            return "NPP_SCALE_RANGE_ERROR";
        case NPP_DATA_TYPE_ERROR:
            return "NPP_DATA_TYPE_ERROR";
        case NPP_OUT_OFF_RANGE_ERROR:
            return "NPP_OUT_OFF_RANGE_ERROR";
        case NPP_DIVIDE_BY_ZERO_ERROR:
            return "NPP_DIVIDE_BY_ZERO_ERROR";
        case NPP_MEMORY_ALLOCATION_ERR:
        case NPP_NULL_POINTER_ERROR:
            return "NPP_NULL_POINTER_ERROR";
        case NPP_RANGE_ERROR:
            return "NPP_RANGE_ERROR";
        case NPP_SIZE_ERROR:
            return "NPP_SIZE_ERROR";
        case NPP_BAD_ARGUMENT_ERROR:
            return "NPP_BAD_ARGUMENT_ERROR";
        case NPP_NO_MEMORY_ERROR:
            return "NPP_NO_MEMORY_ERROR";
        case NPP_NOT_IMPLEMENTED_ERROR:
            return "NPP_NOT_IMPLEMENTED_ERROR";
        case NPP_ERROR:
            return "NPP_ERROR";
        case NPP_ERROR_RESERVED:
            return "NPP_ERROR_RESERVED";

        case NPP_SUCCESS:
            return "NPP_SUCCESS";

        case NPP_NO_OPERATION_WARNING:
            return "NPP_NO_OPERATION_WARNING";
        case NPP_DIVIDE_BY_ZERO_WARNING:
            return "NPP_DIVIDE_BY_ZERO_WARNING";
        case NPP_AFFINE_QUAD_INCORRECT_WARNING:
            return "NPP_AFFINE_QUAD_INCORRECT_WARNING";
        case NPP_WRONG_INTERSECTION_ROI_WARNING:
            return "NPP_WRONG_INTERSECTION_ROI_WARNING";
        case NPP_WRONG_INTERSECTION_QUAD_WARNING:
            return "NPP_WRONG_INTERSECTION_QUAD_WARNING";
        case NPP_DOUBLE_SIZE_WARNING:
            return "NPP_DOUBLE_SIZE_WARNING";

        case NPP_MISALIGNED_DST_ROI_WARNING:
            return "NPP_MISALIGNED_DST_ROI_WARNING";
        default:
            return "UNKNOWN";
    }
}
} // namespace opp::npp

/// <summary>
/// Checks if a NppStatus is NPP_SUCCESS and throws a corresponding NppException if not
/// </summary>
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
inline void __nppSafeCall(NppStatus aErr, const char *aCodeFile, int aLine, const char *aFunction)
{
    if constexpr (NPP_TRHOW_ON_WARNING)
    {
        if (NPP_SUCCESS != aErr)
        {
            throw opp::npp::NppException(aErr, aCodeFile, aLine, aFunction);
        }
    }
    else
    {
        if (aErr < NPP_SUCCESS) // negative codes indicate error, positive warning
        {
            throw opp::npp::NppException(aErr, aCodeFile, aLine, aFunction);
        }
    }
}

/// <summary>
/// Checks if a NppStatus is NPP_SUCCESS and throws a corresponding NppException if not
/// </summary>
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
inline void __nppSafeCall(NppStatus aErr, const std::string &aMessage, const char *aCodeFile, int aLine,
                          const char *aFunction)
{
    if constexpr (NPP_TRHOW_ON_WARNING)
    {
        if (NPP_SUCCESS != aErr)
        {
            throw opp::npp::NppException(aErr, aMessage, aCodeFile, aLine, aFunction);
        }
    }
    else
    {
        if (aErr < NPP_SUCCESS) // negative codes indicate error, positive warning
        {
            throw opp::npp::NppException(aErr, aMessage, aCodeFile, aLine, aFunction);
        }
    }
}

// NOLINTBEGIN --> function like macro, parantheses for "msg"...
#define nppSafeCall(err) __nppSafeCall(err, __FILE__, __LINE__, __PRETTY_FUNCTION__)

#define nppSafeCallExt(err, msg)                                                                                       \
    {                                                                                                                  \
        NppStatus __res = (err);                                                                                       \
        if (__res != NPP_SUCCESS)                                                                                      \
        {                                                                                                              \
            __nppSafeCall(__res, (std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__);        \
        }                                                                                                              \
    }

#define NPPEXCEPTION(msg)                                                                                              \
    (opp::npp::NppException((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

// NOLINTEND
