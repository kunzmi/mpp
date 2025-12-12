#pragma once
#include "../dllexport_npp.h"
#include "image64fC1View.h"
#include "image64fC2View.h"
#include "image64fC3View.h"
#include "image64fC4View.h"
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

class MPPEXPORT_NPP Image64fC1 : public Image64fC1View
{
  public:
    Image64fC1() = delete;
    Image64fC1(int aWidth, int aHeight);
    explicit Image64fC1(const Size2D &aSize);

    ~Image64fC1() override;

    Image64fC1(const Image64fC1 &) = delete;
    Image64fC1(Image64fC1 &&aOther) noexcept;

    Image64fC1 &operator=(const Image64fC1 &) = delete;
    Image64fC1 &operator=(Image64fC1 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image64fC2 : public Image64fC2View
{
  public:
    Image64fC2() = delete;
    Image64fC2(int aWidth, int aHeight);
    explicit Image64fC2(const Size2D &aSize);

    ~Image64fC2() override;

    Image64fC2(const Image64fC2 &) = delete;
    Image64fC2(Image64fC2 &&aOther) noexcept;

    Image64fC2 &operator=(const Image64fC2 &) = delete;
    Image64fC2 &operator=(Image64fC2 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image64fC3 : public Image64fC3View
{
  public:
    Image64fC3() = delete;
    Image64fC3(int aWidth, int aHeight);
    explicit Image64fC3(const Size2D &aSize);

    ~Image64fC3() override;

    Image64fC3(const Image64fC3 &) = delete;
    Image64fC3(Image64fC3 &&aOther) noexcept;

    Image64fC3 &operator=(const Image64fC3 &) = delete;
    Image64fC3 &operator=(Image64fC3 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image64fC4 : public Image64fC4View
{
  public:
    Image64fC4() = delete;
    Image64fC4(int aWidth, int aHeight);
    explicit Image64fC4(const Size2D &aSize);

    ~Image64fC4() override;

    Image64fC4(const Image64fC4 &) = delete;
    Image64fC4(Image64fC4 &&aOther) noexcept;

    Image64fC4 &operator=(const Image64fC4 &) = delete;
    Image64fC4 &operator=(Image64fC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
