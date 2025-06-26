#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "image32sC1View.h"
#include "image32sC2View.h"
#include "image32sC3View.h"
#include "image32sC4View.h"
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

class Image32sC1 : public Image32sC1View
{
  public:
    Image32sC1() = delete;
    Image32sC1(int aWidth, int aHeight);
    explicit Image32sC1(const Size2D &aSize);

    ~Image32sC1();

    Image32sC1(const Image32sC1 &) = delete;
    Image32sC1(Image32sC1 &&aOther) noexcept;

    Image32sC1 &operator=(const Image32sC1 &) = delete;
    Image32sC1 &operator=(Image32sC1 &&aOther) noexcept;
};

class Image32sC2 : public Image32sC2View
{
  public:
    Image32sC2() = delete;
    Image32sC2(int aWidth, int aHeight);
    explicit Image32sC2(const Size2D &aSize);

    ~Image32sC2();

    Image32sC2(const Image32sC2 &) = delete;
    Image32sC2(Image32sC2 &&aOther) noexcept;

    Image32sC2 &operator=(const Image32sC2 &) = delete;
    Image32sC2 &operator=(Image32sC2 &&aOther) noexcept;
};

class Image32sC3 : public Image32sC3View
{
  public:
    Image32sC3() = delete;
    Image32sC3(int aWidth, int aHeight);
    explicit Image32sC3(const Size2D &aSize);

    ~Image32sC3();

    Image32sC3(const Image32sC3 &) = delete;
    Image32sC3(Image32sC3 &&aOther) noexcept;

    Image32sC3 &operator=(const Image32sC3 &) = delete;
    Image32sC3 &operator=(Image32sC3 &&aOther) noexcept;
};

class Image32sC4 : public Image32sC4View
{
  public:
    Image32sC4() = delete;
    Image32sC4(int aWidth, int aHeight);
    explicit Image32sC4(const Size2D &aSize);

    ~Image32sC4();

    Image32sC4(const Image32sC4 &) = delete;
    Image32sC4(Image32sC4 &&aOther) noexcept;

    Image32sC4 &operator=(const Image32sC4 &) = delete;
    Image32sC4 &operator=(Image32sC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
#endif // MPP_ENABLE_NPP_BACKEND
