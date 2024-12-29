#pragma once

#include "image8uC1View.h"
#include "image8uC2View.h"
#include "image8uC3View.h"
#include "image8uC4View.h"
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
};

class Image8uC2 : public Image8uC2View
{
  public:
    Image8uC2() = delete;
    Image8uC2(int aWidth, int aHeight);
    explicit Image8uC2(const Size2D &aSize);

    ~Image8uC2();

    Image8uC2(const Image8uC2 &) = delete;
    Image8uC2(Image8uC2 &&aOther) noexcept;

    Image8uC2 &operator=(const Image8uC2 &) = delete;
    Image8uC2 &operator=(Image8uC2 &&aOther) noexcept;
};

class Image8uC3 : public Image8uC3View
{
  public:
    Image8uC3() = delete;
    Image8uC3(int aWidth, int aHeight);
    explicit Image8uC3(const Size2D &aSize);

    ~Image8uC3();

    Image8uC3(const Image8uC3 &) = delete;
    Image8uC3(Image8uC3 &&aOther) noexcept;

    Image8uC3 &operator=(const Image8uC3 &) = delete;
    Image8uC3 &operator=(Image8uC3 &&aOther) noexcept;
};

class Image8uC4 : public Image8uC4View
{
  public:
    Image8uC4() = delete;
    Image8uC4(int aWidth, int aHeight);
    explicit Image8uC4(const Size2D &aSize);

    ~Image8uC4();

    Image8uC4(const Image8uC4 &) = delete;
    Image8uC4(Image8uC4 &&aOther) noexcept;

    Image8uC4 &operator=(const Image8uC4 &) = delete;
    Image8uC4 &operator=(Image8uC4 &&aOther) noexcept;
};
} // namespace opp::image::npp
