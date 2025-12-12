#pragma once
#include "../dllexport_npp.h"
#include "image32fcC1View.h"
#include "image32fcC2View.h"
#include "image32fcC3View.h"
#include "image32fcC4View.h"
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

class MPPEXPORT_NPP Image32fcC1 : public Image32fcC1View
{
  public:
    Image32fcC1() = delete;
    Image32fcC1(int aWidth, int aHeight);
    explicit Image32fcC1(const Size2D &aSize);

    ~Image32fcC1() override;

    Image32fcC1(const Image32fcC1 &) = delete;
    Image32fcC1(Image32fcC1 &&aOther) noexcept;

    Image32fcC1 &operator=(const Image32fcC1 &) = delete;
    Image32fcC1 &operator=(Image32fcC1 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image32fcC2 : public Image32fcC2View
{
  public:
    Image32fcC2() = delete;
    Image32fcC2(int aWidth, int aHeight);
    explicit Image32fcC2(const Size2D &aSize);

    ~Image32fcC2() override;

    Image32fcC2(const Image32fcC2 &) = delete;
    Image32fcC2(Image32fcC2 &&aOther) noexcept;

    Image32fcC2 &operator=(const Image32fcC2 &) = delete;
    Image32fcC2 &operator=(Image32fcC2 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image32fcC3 : public Image32fcC3View
{
  public:
    Image32fcC3() = delete;
    Image32fcC3(int aWidth, int aHeight);
    explicit Image32fcC3(const Size2D &aSize);

    ~Image32fcC3() override;

    Image32fcC3(const Image32fcC3 &) = delete;
    Image32fcC3(Image32fcC3 &&aOther) noexcept;

    Image32fcC3 &operator=(const Image32fcC3 &) = delete;
    Image32fcC3 &operator=(Image32fcC3 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image32fcC4 : public Image32fcC4View
{
  public:
    Image32fcC4() = delete;
    Image32fcC4(int aWidth, int aHeight);
    explicit Image32fcC4(const Size2D &aSize);

    ~Image32fcC4() override;

    Image32fcC4(const Image32fcC4 &) = delete;
    Image32fcC4(Image32fcC4 &&aOther) noexcept;

    Image32fcC4 &operator=(const Image32fcC4 &) = delete;
    Image32fcC4 &operator=(Image32fcC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
