#include "fileIOException.h"
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace opp::fileIO
{
FileIOException::FileIOException(const std::filesystem::path &aFileName, const std::string &aMessage,
                                 [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                                 [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "FileIOException for file: '" << aFileName.generic_string() << "'" << std::endl
       << "Error message: " << aMessage;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "FileIOException for file: '" << aFileName.generic_string() << "'" << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}
} // namespace opp::fileIO
