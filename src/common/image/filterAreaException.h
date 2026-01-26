#pragma once
#include "../dllexport_common.h"
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/filterArea.h>
#include <common/numberTypes.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace mpp::image
{
/// <summary>
/// FilterAreaException is thrown when a FilterArea is not valid.
/// </summary>
class MPPEXPORT_COMMON FilterAreaException : public MPPException
{
  public:
    FilterAreaException(const FilterArea &aArea, const std::filesystem::path &aCodeFileName, int aLineNumber,
                        const std::string &aFunctionName);
    ~FilterAreaException() noexcept override = default;

    FilterAreaException(FilterAreaException &&) noexcept        = default;
    FilterAreaException(const FilterAreaException &)            = default;
    FilterAreaException &operator=(const FilterAreaException &) = delete;
    FilterAreaException &operator=(FilterAreaException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::FilterArea;
    }
};
} // namespace mpp::image

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

/// <summary>
/// Checks if a pointer is nullptr and throws NullPtrException if it is
/// </summary>
inline void __checkFilterArea(const mpp::image::FilterArea &aArea, const char *aCodeFile, int aLine,
                              const char *aFunction)
{
    if (!aArea.CheckIfValid())
    {
        throw mpp::image::FilterAreaException(aArea, aCodeFile, aLine, aFunction);
    }
}
#define checkFilterArea(area) __checkFilterArea(area, __FILE__, __LINE__, __PRETTY_FUNCTION__);

// NOLINTEND
