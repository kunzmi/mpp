#include "fileIOException.h"
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace opp::fileIO
{
FileIOException::FileIOException(const std::filesystem::path &aFileName, const std::string &aMessage,
                                 const std::filesystem::path &aCodeFileName, int aLineNumber,
                                 const std::string &aFunctionName)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "FileIOException for file: '" << aFileName.generic_string() << "'" << std::endl
       << "Error message: " << aMessage;
#else
    std::stringstream ss;
    ss << "Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber << std::endl
       << "FileIOException for file: '" << aFileName.generic_string() << "'" << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}
} // namespace opp::fileIO
