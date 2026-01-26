#include "pitchException.h"
#include <cstddef>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace mpp::image
{

PitchException::PitchException(size_t aPitch, int aWidth, size_t aTypeSize,
                               [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                               [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "Image pitch " << aPitch << " is smaller than image width x pixel size in bytes: " << aWidth << " x "
       << aTypeSize << " = " << static_cast<size_t>(aWidth) * aTypeSize << "." << std::endl;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Pitch-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Image pitch " << aPitch << " is smaller than image width x pixel size in bytes: " << aWidth << " x "
       << aTypeSize << " = " << static_cast<size_t>(aWidth) * aTypeSize << "." << std::endl;
#endif

    What() = ss.str();
}

PitchException::PitchException(const std::string &aName, size_t aPitch, int aWidth, size_t aTypeSize,
                               [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                               [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "Pitch for " << aName << " (" << aPitch << ") is smaller than image width x pixel size in bytes: " << aWidth
       << " x " << aTypeSize << " = " << static_cast<size_t>(aWidth) * aTypeSize << "." << std::endl;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Pitch-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Pitch for " << aName << " (" << aPitch << ") is smaller than image width x pixel size in bytes: " << aWidth
       << " x " << aTypeSize << " = " << static_cast<size_t>(aWidth) * aTypeSize << "." << std::endl;
#endif

    What() = ss.str();
}

PitchException::PitchException(size_t aPitch, int aMinimumPitch,
                               [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                               [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "Pitch (or step) " << aPitch << " [bytes] is not a multiple of the required minimum pitch " << aMinimumPitch
       << " [bytes] that is required for this operation." << std::endl;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Pitch-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Pitch (or step) " << aPitch << " [bytes] is not a multiple of the required minimum pitch " << aMinimumPitch
       << " [bytes] that is required for this operation." << std::endl;
#endif

    What() = ss.str();
}

} // namespace mpp::image