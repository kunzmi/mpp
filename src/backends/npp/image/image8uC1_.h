#pragma once

#include "image8uC1View.h"
#include <backends/npp/nppException.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>

namespace opp::image::npp
{

class Image8uC1 : public Image8uC1View
{

  public:
    Image8uC1() = delete;
    Image8uC1(int aWidth, int aHeight);
    explicit Image8uC1(const Size2D &aSize);

    ~Image8uC1();

    Image8uC1(const Image8uC1 &) = delete;
    Image8uC1(Image8uC1 &&aOther) noexcept;

    Image8uC1 &operator=(const Image8uC1 &) = delete;
    Image8uC1 &operator=(Image8uC1 &&aOther) noexcept;


    void Add(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx);
    void Div(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx);
    void Mul(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx);
    void Sub(const Image8uC1 &aSrc1, const Image8uC1 &aSrc2, int aScaleFactor, const NppStreamContext &aStreamCtx);

};
} // namespace opp::image::npp
