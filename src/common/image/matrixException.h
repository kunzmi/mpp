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
/// MatrixException is thrown when a matrix operation is not valid.
/// </summary>
class MPPEXPORT_COMMON MatrixException : public MPPException
{
  public:
    MatrixException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                    const std::string &aFunctionName);
    ~MatrixException() noexcept override = default;

    MatrixException(MatrixException &&) noexcept        = default;
    MatrixException(const MatrixException &)            = default;
    MatrixException &operator=(const MatrixException &) = delete;
    MatrixException &operator=(MatrixException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::Matrix;
    }
};
} // namespace mpp::image

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

#define MATRIXEXCEPTION(msg)                                                                                           \
    (mpp::image::MatrixException((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

// NOLINTEND
