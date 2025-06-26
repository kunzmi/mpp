#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "image32uC1View.h"
#include "image32uC2View.h"
#include "image32uC3View.h"
#include "image32uC4View.h"
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

class Image32uC1 : public Image32uC1View
{
  public:
    Image32uC1() = delete;
    Image32uC1(int aWidth, int aHeight);
    explicit Image32uC1(const Size2D &aSize);

    ~Image32uC1();

    Image32uC1(const Image32uC1 &) = delete;
    Image32uC1(Image32uC1 &&aOther) noexcept;

    Image32uC1 &operator=(const Image32uC1 &) = delete;
    Image32uC1 &operator=(Image32uC1 &&aOther) noexcept;
};

class Image32uC2 : public Image32uC2View
{
  public:
    Image32uC2() = delete;
    Image32uC2(int aWidth, int aHeight);
    explicit Image32uC2(const Size2D &aSize);

    ~Image32uC2();

    Image32uC2(const Image32uC2 &) = delete;
    Image32uC2(Image32uC2 &&aOther) noexcept;

    Image32uC2 &operator=(const Image32uC2 &) = delete;
    Image32uC2 &operator=(Image32uC2 &&aOther) noexcept;
};

class Image32uC3 : public Image32uC3View
{
  public:
    Image32uC3() = delete;
    Image32uC3(int aWidth, int aHeight);
    explicit Image32uC3(const Size2D &aSize);

    ~Image32uC3();

    Image32uC3(const Image32uC3 &) = delete;
    Image32uC3(Image32uC3 &&aOther) noexcept;

    Image32uC3 &operator=(const Image32uC3 &) = delete;
    Image32uC3 &operator=(Image32uC3 &&aOther) noexcept;
};

class Image32uC4 : public Image32uC4View
{
  public:
    Image32uC4() = delete;
    Image32uC4(int aWidth, int aHeight);
    explicit Image32uC4(const Size2D &aSize);

    ~Image32uC4();

    Image32uC4(const Image32uC4 &) = delete;
    Image32uC4(Image32uC4 &&aOther) noexcept;

    Image32uC4 &operator=(const Image32uC4 &) = delete;
    Image32uC4 &operator=(Image32uC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
#endif // MPP_ENABLE_NPP_BACKEND
