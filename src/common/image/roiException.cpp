#include "roiException.h"
#include <common/exception.h>
#include <common/image/roi.h>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace mpp::image
{

RoiException::RoiException(const Roi &aRoi, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                           [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "'" << aRoi << std::endl;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "ROI-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << aRoi << std::endl;
#endif

    What() = ss.str();
}

RoiException::RoiException(const Roi &aRoi, const std::string &aMessage,
                           [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                           [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
    : MPPException(aMessage)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << aRoi << std::endl << "Additional info: " << aMessage;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "ROI-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << aRoi << std::endl
       << "Additional info: " << aMessage;
#endif

    What() = ss.str();
}

RoiException::RoiException(const std::string &aMessage, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
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
    ss << "ROI-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}

} // namespace mpp::image