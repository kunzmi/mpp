#include "roiException.h"
#include <common/exception.h>
#include <common/image/roi.h>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace opp::image
{

RoiException::RoiException(const Roi &aRoi, const std::filesystem::path &aCodeFileName, int aLineNumber,
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
    ss << "'" << aRoi << std::endl;
#else
    std::stringstream ss;
    ss << "ROI-Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << aRoi << std::endl;
#endif

    What() = ss.str();
}

RoiException::RoiException(const Roi &aRoi, const std::string &aMessage, const std::filesystem::path &aCodeFileName,
                           int aLineNumber, const std::string &aFunctionName)
    : OPPException(aMessage)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << aRoi << std::endl << "Additional info: " << aMessage;
#else
    std::stringstream ss;
    ss << "ROI-Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << aRoi << std::endl
       << "Additional info: " << aMessage;
#endif

    What() = ss.str();
}

RoiException::RoiException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                           const std::string &aFunctionName)
    : OPPException(aMessage)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "Error message: " << aMessage;
#else
    std::stringstream ss;
    ss << "ROI-Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}

} // namespace opp::image