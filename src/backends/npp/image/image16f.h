#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "image16fC1View.h"
#include "image16fC2View.h"
#include "image16fC3View.h"
#include "image16fC4View.h"
#include <backends/cuda/devVarView.h>
#include <backends/npp/nppException.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <nppdefs.h>

namespace mpp::image::npp
{

class Image16fC1 : public Image16fC1View
{
  public:
    Image16fC1() = delete;
    Image16fC1(int aWidth, int aHeight);
    explicit Image16fC1(const Size2D &aSize);

    ~Image16fC1();

    Image16fC1(const Image16fC1 &) = delete;
    Image16fC1(Image16fC1 &&aOther) noexcept;

    Image16fC1 &operator=(const Image16fC1 &) = delete;
    Image16fC1 &operator=(Image16fC1 &&aOther) noexcept;
};

class Image16fC2 : public Image16fC2View
{
  public:
    Image16fC2() = delete;
    Image16fC2(int aWidth, int aHeight);
    explicit Image16fC2(const Size2D &aSize);

    ~Image16fC2();

    Image16fC2(const Image16fC2 &) = delete;
    Image16fC2(Image16fC2 &&aOther) noexcept;

    Image16fC2 &operator=(const Image16fC2 &) = delete;
    Image16fC2 &operator=(Image16fC2 &&aOther) noexcept;
};

class Image16fC3 : public Image16fC3View
{
  public:
    Image16fC3() = delete;
    Image16fC3(int aWidth, int aHeight);
    explicit Image16fC3(const Size2D &aSize);

    ~Image16fC3();

    Image16fC3(const Image16fC3 &) = delete;
    Image16fC3(Image16fC3 &&aOther) noexcept;

    Image16fC3 &operator=(const Image16fC3 &) = delete;
    Image16fC3 &operator=(Image16fC3 &&aOther) noexcept;
};

class Image16fC4 : public Image16fC4View
{
  public:
    Image16fC4() = delete;
    Image16fC4(int aWidth, int aHeight);
    explicit Image16fC4(const Size2D &aSize);

    ~Image16fC4();

    Image16fC4(const Image16fC4 &) = delete;
    Image16fC4(Image16fC4 &&aOther) noexcept;

    Image16fC4 &operator=(const Image16fC4 &) = delete;
    Image16fC4 &operator=(Image16fC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
#endif // MPP_ENABLE_NPP_BACKEND
