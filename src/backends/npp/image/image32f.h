#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "../dllexport_npp.h"
#include "image32fC1View.h"
#include "image32fC2View.h"
#include "image32fC3View.h"
#include "image32fC4View.h"
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

class MPPEXPORT_NPP Image32fC1 : public Image32fC1View
{
  public:
    Image32fC1() = delete;
    Image32fC1(int aWidth, int aHeight);
    explicit Image32fC1(const Size2D &aSize);

    ~Image32fC1();

    Image32fC1(const Image32fC1 &) = delete;
    Image32fC1(Image32fC1 &&aOther) noexcept;

    Image32fC1 &operator=(const Image32fC1 &) = delete;
    Image32fC1 &operator=(Image32fC1 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image32fC2 : public Image32fC2View
{
  public:
    Image32fC2() = delete;
    Image32fC2(int aWidth, int aHeight);
    explicit Image32fC2(const Size2D &aSize);

    ~Image32fC2();

    Image32fC2(const Image32fC2 &) = delete;
    Image32fC2(Image32fC2 &&aOther) noexcept;

    Image32fC2 &operator=(const Image32fC2 &) = delete;
    Image32fC2 &operator=(Image32fC2 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image32fC3 : public Image32fC3View
{
  public:
    Image32fC3() = delete;
    Image32fC3(int aWidth, int aHeight);
    explicit Image32fC3(const Size2D &aSize);

    ~Image32fC3();

    Image32fC3(const Image32fC3 &) = delete;
    Image32fC3(Image32fC3 &&aOther) noexcept;

    Image32fC3 &operator=(const Image32fC3 &) = delete;
    Image32fC3 &operator=(Image32fC3 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image32fC4 : public Image32fC4View
{
  public:
    Image32fC4() = delete;
    Image32fC4(int aWidth, int aHeight);
    explicit Image32fC4(const Size2D &aSize);

    ~Image32fC4();

    Image32fC4(const Image32fC4 &) = delete;
    Image32fC4(Image32fC4 &&aOther) noexcept;

    Image32fC4 &operator=(const Image32fC4 &) = delete;
    Image32fC4 &operator=(Image32fC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
#endif // MPP_ENABLE_NPP_BACKEND
