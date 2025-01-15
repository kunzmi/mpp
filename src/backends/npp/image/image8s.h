#pragma once
#include <common/moduleEnabler.h>
#if OPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "image8sC1View.h"
#include "image8sC2View.h"
#include "image8sC3View.h"
#include "image8sC4View.h"
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

class Image8sC1 : public Image8sC1View
{
  public:
    Image8sC1() = delete;
    Image8sC1(int aWidth, int aHeight);
    explicit Image8sC1(const Size2D &aSize);

    ~Image8sC1();

    Image8sC1(const Image8sC1 &) = delete;
    Image8sC1(Image8sC1 &&aOther) noexcept;

    Image8sC1 &operator=(const Image8sC1 &) = delete;
    Image8sC1 &operator=(Image8sC1 &&aOther) noexcept;
};

class Image8sC2 : public Image8sC2View
{
  public:
    Image8sC2() = delete;
    Image8sC2(int aWidth, int aHeight);
    explicit Image8sC2(const Size2D &aSize);

    ~Image8sC2();

    Image8sC2(const Image8sC2 &) = delete;
    Image8sC2(Image8sC2 &&aOther) noexcept;

    Image8sC2 &operator=(const Image8sC2 &) = delete;
    Image8sC2 &operator=(Image8sC2 &&aOther) noexcept;
};

class Image8sC3 : public Image8sC3View
{
  public:
    Image8sC3() = delete;
    Image8sC3(int aWidth, int aHeight);
    explicit Image8sC3(const Size2D &aSize);

    ~Image8sC3();

    Image8sC3(const Image8sC3 &) = delete;
    Image8sC3(Image8sC3 &&aOther) noexcept;

    Image8sC3 &operator=(const Image8sC3 &) = delete;
    Image8sC3 &operator=(Image8sC3 &&aOther) noexcept;
};

class Image8sC4 : public Image8sC4View
{
  public:
    Image8sC4() = delete;
    Image8sC4(int aWidth, int aHeight);
    explicit Image8sC4(const Size2D &aSize);

    ~Image8sC4();

    Image8sC4(const Image8sC4 &) = delete;
    Image8sC4(Image8sC4 &&aOther) noexcept;

    Image8sC4 &operator=(const Image8sC4 &) = delete;
    Image8sC4 &operator=(Image8sC4 &&aOther) noexcept;
};
} // namespace opp::image::npp
#endif // OPP_ENABLE_NPP_BACKEND
