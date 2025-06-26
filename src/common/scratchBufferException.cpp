#include "scratchBufferException.h"
#include <cstddef>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace mpp
{
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
ScratchBufferException::ScratchBufferException(size_t aRequiredSize, size_t aProvidedSize, const std::string &aMessage,
                                               [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                                               [[maybe_unused]] int aLineNumber,
                                               [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "ScratchBuffer-Error - required buffer size [bytes]: " << aRequiredSize << std::endl
       << "Provided buffer size [bytes]: " << aProvidedSize << std::endl;
    if (!aMessage.empty())
    {
        ss << "Error message: " << aMessage << std::endl;
    }
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "ScratchBuffer-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ "
       << aLineNumber << std::endl
       << "Required buffer size [bytes]: " << aRequiredSize << std::endl
       << "Provided buffer size [bytes]: " << aProvidedSize << std::endl;
    if (!aMessage.empty())
    {
        ss << "Error message: " << aMessage << std::endl;
    }
#endif

    What() = ss.str();
}

} // namespace mpp