#include "filterArea.h"
#include "filterAreaException.h"
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace mpp::image
{

FilterAreaException::FilterAreaException(const FilterArea &aArea,
                                         [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                                         [[maybe_unused]] int aLineNumber,
                                         [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "Error message: FilterArea " << aArea << " is not valid.";
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "FilterArea-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ "
       << aLineNumber << std::endl
       << "Error message: FilterArea " << aArea << " is not valid.";
#endif

    What() = ss.str();
}

} // namespace mpp::image