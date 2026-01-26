#pragma once
#include "../dllexport_common.h"
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/imageViewBase.h>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace mpp::image
{
/// <summary>
/// PitchException is thrown when an image pitch is smaller than an image line byte size or that the pitch is not a
/// multiple of the minimum pitch.
/// </summary>
class MPPEXPORT_COMMON PitchException : public MPPException
{
  public:
    PitchException(size_t aPitch, int aWidth, size_t aTypeSize, const std::filesystem::path &aCodeFileName,
                   int aLineNumber, const std::string &aFunctionName);
    PitchException(const std::string &aName, size_t aPitch, int aWidth, size_t aTypeSize,
                   const std::filesystem::path &aCodeFileName, int aLineNumber, const std::string &aFunctionName);
    PitchException(size_t aPitch, int aMinimumPitch, const std::filesystem::path &aCodeFileName, int aLineNumber,
                   const std::string &aFunctionName);
    ~PitchException() noexcept override = default;

    PitchException(PitchException &&) noexcept        = default;
    PitchException(const PitchException &)            = default;
    PitchException &operator=(const PitchException &) = delete;
    PitchException &operator=(PitchException &&)      = delete;
};
} // namespace mpp::image

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

/// <summary>
/// Checks if Pitch at least as large as an image line in bytes
/// </summary>
template <typename T>
void __checkPitch(const std::string &aName, const mpp::image::ImageViewBase<T> &aImg, const char *aCodeFile, int aLine,
                  const char *aFunction)
{
    if (aImg.Pointer() != nullptr && aImg.Pitch() < aImg.PixelSizeInBytes * static_cast<size_t>(aImg.Width()))
    {
        throw mpp::image::PitchException(aName == "*this" ? "first source image" : aName, aImg.Pitch(), aImg.Width(),
                                         aImg.PixelSizeInBytes, aCodeFile, aLine, aFunction);
    }
}

inline void __checkPitchIsMultiple(size_t aPitch, int aMinimumPitch, size_t aTupelSize, const char *aCodeFile,
                                   int aLine, const char *aFunction)
{
    if (aTupelSize != 1)
    {
        if (aPitch % static_cast<size_t>(aMinimumPitch) != 0)
        {
            throw mpp::image::PitchException(aPitch, aMinimumPitch, aCodeFile, aLine, aFunction);
        }
    }
}

#define checkPitch(aImg) __checkPitch(#aImg, aImg, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define checkPitchIsMultiple(aPitch, aMinPitch, aTupelSize)                                                            \
    __checkPitchIsMultiple(aPitch, aMinPitch, aTupelSize, __FILE__, __LINE__, __PRETTY_FUNCTION__)

// NOLINTEND
