#include "errorMessageSingleton.h"
#include "exception.h"
#include <string>

namespace mpp
{

namespace
{
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local std::string tlSingletonLastErrorMessage{};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local int tlSingletonLastErrorCode{0};
} // namespace

const std::string &ErrorMessageSingleton::GetLastErrorMessage()
{
    return tlSingletonLastErrorMessage;
}

int ErrorMessageSingleton::GetLastErrorCode()
{
    return tlSingletonLastErrorCode;
}

void ErrorMessageSingleton::SetLastError(const std::string &aMessage, ExceptionCode aCode)
{
    tlSingletonLastErrorMessage = aMessage;
    tlSingletonLastErrorCode    = static_cast<int>(aCode);
}

} // namespace mpp
