#pragma once
#include <backends/cuda/devVarView.h>
#include <common/exception.h>
#include <common/image/imageViewBase.h>
#include <common/image/pitchException.h>

// NOLINTBEGIN --> function like macro, parantheses for "msg"...
/// <summary>
/// Checks if an image is nullptr and throws NullPtrException if it is
/// </summary>
template <typename T>
void __checkNullptr(const mpp::image::ImageViewBase<T> &aImg, const std::string &aName, const char *aCodeFile,
                    int aLine, const char *aFunction)
{
    if (aImg.Pointer() == nullptr)
    {
        throw mpp::NullPtrException(aName == "*this" ? "first source image" : aName, aCodeFile, aLine, aFunction);
    }
}

#define validateImage(image)                                                                                           \
    checkPitch(image);                                                                                                 \
    checkNullptr(image);

// NOLINTEND