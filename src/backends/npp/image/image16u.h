#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include "../dllexport_npp.h"
#include "image16uC1View.h"
#include "image16uC2View.h"
#include "image16uC3View.h"
#include "image16uC4View.h"
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

class MPPEXPORT_NPP Image16uC1 : public Image16uC1View
{
  public:
    Image16uC1() = delete;
    Image16uC1(int aWidth, int aHeight);
    explicit Image16uC1(const Size2D &aSize);

    ~Image16uC1();

    Image16uC1(const Image16uC1 &) = delete;
    Image16uC1(Image16uC1 &&aOther) noexcept;

    Image16uC1 &operator=(const Image16uC1 &) = delete;
    Image16uC1 &operator=(Image16uC1 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image16uC2 : public Image16uC2View
{
  public:
    Image16uC2() = delete;
    Image16uC2(int aWidth, int aHeight);
    explicit Image16uC2(const Size2D &aSize);

    ~Image16uC2();

    Image16uC2(const Image16uC2 &) = delete;
    Image16uC2(Image16uC2 &&aOther) noexcept;

    Image16uC2 &operator=(const Image16uC2 &) = delete;
    Image16uC2 &operator=(Image16uC2 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image16uC3 : public Image16uC3View
{
  public:
    Image16uC3() = delete;
    Image16uC3(int aWidth, int aHeight);
    explicit Image16uC3(const Size2D &aSize);

    ~Image16uC3();

    Image16uC3(const Image16uC3 &) = delete;
    Image16uC3(Image16uC3 &&aOther) noexcept;

    Image16uC3 &operator=(const Image16uC3 &) = delete;
    Image16uC3 &operator=(Image16uC3 &&aOther) noexcept;
};

class MPPEXPORT_NPP Image16uC4 : public Image16uC4View
{
  public:
    Image16uC4() = delete;
    Image16uC4(int aWidth, int aHeight);
    explicit Image16uC4(const Size2D &aSize);

    ~Image16uC4();

    Image16uC4(const Image16uC4 &) = delete;
    Image16uC4(Image16uC4 &&aOther) noexcept;

    Image16uC4 &operator=(const Image16uC4 &) = delete;
    Image16uC4 &operator=(Image16uC4 &&aOther) noexcept;
};
} // namespace mpp::image::npp
#endif // MPP_ENABLE_NPP_BACKEND
