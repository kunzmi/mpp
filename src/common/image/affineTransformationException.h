#pragma once
#include "../dllexport_common.h"
#include <common/defines.h>
#include <common/exception.h>
#include <common/numberTypes.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace mpp::image
{
/// <summary>
/// AffineTransformationException is thrown when an operation on an affine transformation is not valid.
/// </summary>
class MPPEXPORT_COMMON AffineTransformationException : public MPPException
{
  public:
    AffineTransformationException(const std::string &aMessage, const std::filesystem::path &aCodeFileName,
                                  int aLineNumber, const std::string &aFunctionName);
    ~AffineTransformationException() noexcept override = default;

    AffineTransformationException(AffineTransformationException &&) noexcept        = default;
    AffineTransformationException(const AffineTransformationException &)            = default;
    AffineTransformationException &operator=(const AffineTransformationException &) = delete;
    AffineTransformationException &operator=(AffineTransformationException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::AffineTransformation;
    }
};
} // namespace mpp::image

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

#define AFFINETRANSFORMATIONEXCEPTION(msg)                                                                             \
    (mpp::image::AffineTransformationException((std::ostringstream() << msg).str(), __FILE__, __LINE__,                \
                                               __PRETTY_FUNCTION__))

// NOLINTEND
