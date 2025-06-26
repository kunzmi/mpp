#pragma once
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/roi.h>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace mpp::image
{
/// <summary>
/// RoiException is thrown when an operation exceeds the dimension of a given ROI.
/// </summary>
class RoiException : public MPPException
{
  public:
    RoiException(const Roi &aRoi, const std::filesystem::path &aCodeFileName, int aLineNumber,
                 const std::string &aFunctionName);
    RoiException(const Roi &aRoi, const std::string &aMessage, const std::filesystem::path &aCodeFileName,
                 int aLineNumber, const std::string &aFunctionName);
    RoiException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                 const std::string &aFunctionName);
    ~RoiException() noexcept override = default;

    RoiException(RoiException &&)                 = default;
    RoiException(const RoiException &)            = default;
    RoiException &operator=(const RoiException &) = delete;
    RoiException &operator=(RoiException &&)      = delete;
};
} // namespace mpp::image

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

/// <summary>
/// Checks if ROI1 is fully inside ROI2 and throws a RoiException if not
/// </summary>
inline void __checkRoiIsInRoi(const mpp::image::Roi &aRoi1, const mpp::image::Roi &aRoi2, const char *aCodeFile,
                              int aLine, const char *aFunction)
{
    if (aRoi1.Union(aRoi2) != aRoi2)
    {
        std::stringstream ss;
        ss << "The requested " << aRoi1 << " exceeds the " << aRoi2;
        throw mpp::image::RoiException(ss.str(), aCodeFile, aLine, aFunction);
    }
}

/// <summary>
/// Checks if both provided sizes are equal and throws a RoiException if not
/// </summary>
inline void __checkSameSize(const mpp::image::Size2D &aSize1, const mpp::image::Size2D &aSize2, const char *aCodeFile,
                            int aLine, const char *aFunction)
{
    if (aSize1 != aSize2)
    {
        std::stringstream ss;
        ss << "Both ROI sizes must be equal but first size is " << aSize1 << " other size is " << aSize2 << ".";
        throw mpp::image::RoiException(ss.str(), aCodeFile, aLine, aFunction);
    }
}

/// <summary>
/// Checks if both provided sizes are equal and throws a RoiException if not
/// </summary>
inline void __checkSameSize(const mpp::image::Roi &aRoi1, const mpp::image::Roi &aRoi2, const char *aCodeFile,
                            int aLine, const char *aFunction)
{
    if (aRoi1.Size() != aRoi2.Size())
    {
        std::stringstream ss;
        ss << "Both ROI sizes must be equal but first size is " << aRoi1.Size() << " other size is " << aRoi2.Size()
           << ".";
        throw mpp::image::RoiException(ss.str(), aCodeFile, aLine, aFunction);
    }
}

#define checkRoiIsInRoi(aRoi1, aRoi2) __checkRoiIsInRoi(aRoi1, aRoi2, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define checkSameSize(aSize1, aSize2) __checkSameSize(aSize1, aSize2, __FILE__, __LINE__, __PRETTY_FUNCTION__)

#define ROIEXCEPTION(msg)                                                                                              \
    (mpp::image::RoiException((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

// NOLINTEND
