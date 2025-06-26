#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "image16scC1View.h"
#include "image16scC2View.h"
#include "image16scC3View.h"
#include "image16scC4View.h"
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

class Image16scC1 : public Image16scC1View
{
  public:
    Image16scC1() = delete;
    Image16scC1(int aWidth, int aHeight);
    explicit Image16scC1(const Size2D &aSize);

    ~Image16scC1();

    Image16scC1(const Image16scC1 &) = delete;
    Image16scC1(Image16scC1 &&aOther) noexcept;

    Image16scC1 &operator=(const Image16scC1 &) = delete;
    Image16scC1 &operator=(Image16scC1 &&aOther) noexcept;
};

class Image16scC2 : public Image16scC2View
{
  public:
    Image16scC2() = delete;
    Image16scC2(int aWidth, int aHeight);
    explicit Image16scC2(const Size2D &aSize);

    ~Image16scC2();

    Image16scC2(const Image16scC2 &) = delete;
    Image16scC2(Image16scC2 &&aOther) noexcept;

    Image16scC2 &operator=(const Image16scC2 &) = delete;
    Image16scC2 &operator=(Image16scC2 &&aOther) noexcept;
};

class Image16scC3 : public Image16scC3View
{
  public:
    Image16scC3() = delete;
    Image16scC3(int aWidth, int aHeight);
    explicit Image16scC3(const Size2D &aSize);

    ~Image16scC3();

    Image16scC3(const Image16scC3 &) = delete;
    Image16scC3(Image16scC3 &&aOther) noexcept;

    Image16scC3 &operator=(const Image16scC3 &) = delete;
    Image16scC3 &operator=(Image16scC3 &&aOther) noexcept;
};

class Image16scC4 : public Image16scC4View
{
  public:
    Image16scC4() = delete;
    Image16scC4(int aWidth, int aHeight);
    explicit Image16scC4(const Size2D &aSize);

    ~Image16scC4();

    Image16scC4(const Image16scC4 &) = delete;
    Image16scC4(Image16scC4 &&aOther) noexcept;

    Image16scC4 &operator=(const Image16scC4 &) = delete;
    Image16scC4 &operator=(Image16scC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
#endif // MPP_ENABLE_NPP_BACKEND
