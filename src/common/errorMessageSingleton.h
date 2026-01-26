#pragma once
#include "dllexport_common.h"
#include <common/defines.h>
#include <common/exception.h>
#include <string>

namespace mpp
{

/// <summary>
/// For use in C-API only
/// </summary>
class MPPEXPORT_COMMON ErrorMessageSingleton
{
  public:
    /// <summary>
    /// Gets the last ErrorMessage that is associated with the current CPU thread.
    /// </summary>
    [[nodiscard]] static const std::string &GetLastErrorMessage();

    /// <summary>
    /// Gets the last ErrorCode that is associated with the current CPU thread.
    /// </summary>
    [[nodiscard]] static int GetLastErrorCode();

    /// <summary>
    /// Sets the latest error code and error message for this CPU thread.
    /// </summary>
    /// <returns></returns>
    static void SetLastError(const std::string &aMessage, ExceptionCode aCode);
};
} // namespace mpp