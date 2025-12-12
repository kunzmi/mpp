#pragma once
#include "../dllexport_npp.h"
#include "image16sC1View.h"
#include "image16sC2View.h"
#include "image16sC3View.h"
#include "image16sC4View.h"
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

class MPPEXPORT_NPP Image16sC1 : public Image16sC1View
{
  public:
    Image16sC1() = delete;
    Image16sC1(int aWidth, int aHeight);
    explicit Image16sC1(const Size2D &aSize);

    ~Image16sC1() override;

    Image16sC1(const Image16sC1 &) = delete;
    Image16sC1(Image16sC1 &&aOther) noexcept;

    Image16sC1 &operator=(const Image16sC1 &) = delete;
    Image16sC1 &operator=(Image16sC1 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image16sC2 : public Image16sC2View
{
  public:
    Image16sC2() = delete;
    Image16sC2(int aWidth, int aHeight);
    explicit Image16sC2(const Size2D &aSize);

    ~Image16sC2() override;

    Image16sC2(const Image16sC2 &) = delete;
    Image16sC2(Image16sC2 &&aOther) noexcept;

    Image16sC2 &operator=(const Image16sC2 &) = delete;
    Image16sC2 &operator=(Image16sC2 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image16sC3 : public Image16sC3View
{
  public:
    Image16sC3() = delete;
    Image16sC3(int aWidth, int aHeight);
    explicit Image16sC3(const Size2D &aSize);

    ~Image16sC3() override;

    Image16sC3(const Image16sC3 &) = delete;
    Image16sC3(Image16sC3 &&aOther) noexcept;

    Image16sC3 &operator=(const Image16sC3 &) = delete;
    Image16sC3 &operator=(Image16sC3 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image16sC4 : public Image16sC4View
{
  public:
    Image16sC4() = delete;
    Image16sC4(int aWidth, int aHeight);
    explicit Image16sC4(const Size2D &aSize);

    ~Image16sC4() override;

    Image16sC4(const Image16sC4 &) = delete;
    Image16sC4(Image16sC4 &&aOther) noexcept;

    Image16sC4 &operator=(const Image16sC4 &) = delete;
    Image16sC4 &operator=(Image16sC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
