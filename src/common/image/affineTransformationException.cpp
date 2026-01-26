#include "affineTransformationException.h"
#include <common/exception.h>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace mpp::image
{

AffineTransformationException::AffineTransformationException(
    const std::string &aMessage, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
    [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
    : MPPException(aMessage)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "Error message: " << aMessage;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "AffineTransformation-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ "
       << aLineNumber << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}

} // namespace mpp::image