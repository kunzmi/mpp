#pragma once
#include <common/moduleEnabler.h>
#if OPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "image32scC1View.h"
#include "image32scC2View.h"
#include "image32scC3View.h"
#include "image32scC4View.h"
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

class Image32scC1 : public Image32scC1View
{
  public:
    Image32scC1() = delete;
    Image32scC1(int aWidth, int aHeight);
    explicit Image32scC1(const Size2D &aSize);

    ~Image32scC1();

    Image32scC1(const Image32scC1 &) = delete;
    Image32scC1(Image32scC1 &&aOther) noexcept;

    Image32scC1 &operator=(const Image32scC1 &) = delete;
    Image32scC1 &operator=(Image32scC1 &&aOther) noexcept;
};

class Image32scC2 : public Image32scC2View
{
  public:
    Image32scC2() = delete;
    Image32scC2(int aWidth, int aHeight);
    explicit Image32scC2(const Size2D &aSize);

    ~Image32scC2();

    Image32scC2(const Image32scC2 &) = delete;
    Image32scC2(Image32scC2 &&aOther) noexcept;

    Image32scC2 &operator=(const Image32scC2 &) = delete;
    Image32scC2 &operator=(Image32scC2 &&aOther) noexcept;
};

class Image32scC3 : public Image32scC3View
{
  public:
    Image32scC3() = delete;
    Image32scC3(int aWidth, int aHeight);
    explicit Image32scC3(const Size2D &aSize);

    ~Image32scC3();

    Image32scC3(const Image32scC3 &) = delete;
    Image32scC3(Image32scC3 &&aOther) noexcept;

    Image32scC3 &operator=(const Image32scC3 &) = delete;
    Image32scC3 &operator=(Image32scC3 &&aOther) noexcept;
};

class Image32scC4 : public Image32scC4View
{
  public:
    Image32scC4() = delete;
    Image32scC4(int aWidth, int aHeight);
    explicit Image32scC4(const Size2D &aSize);

    ~Image32scC4();

    Image32scC4(const Image32scC4 &) = delete;
    Image32scC4(Image32scC4 &&aOther) noexcept;

    Image32scC4 &operator=(const Image32scC4 &) = delete;
    Image32scC4 &operator=(Image32scC4 &&aOther) noexcept;
};
} // namespace opp::image::npp
#endif // OPP_ENABLE_NPP_BACKEND
