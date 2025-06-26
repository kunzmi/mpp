#pragma once
#include "testBase.h"
#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
// #include <backends/npp/image/image.h>
#include <backends/npp/image/imageView.h>
#include <common/image/pixelTypes.h>
#include <nppcore.h>
#include <nppdefs.h>

namespace opp
{
using namespace opp::cuda;
using namespace opp::image;
// using namespace opp::image::cuda;

class TestNppBase : public TestBase
{
  public:
    TestNppBase(size_t aIterations, size_t aRepeats) : TestBase(aIterations, aRepeats)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
    }
    virtual ~TestNppBase() = default;

    TestNppBase(const TestNppBase &)     = default;
    TestNppBase(TestNppBase &&) noexcept = default;

    TestNppBase &operator=(const TestNppBase &)     = default;
    TestNppBase &operator=(TestNppBase &&) noexcept = default;

  protected:
    NppStreamContext ctx;

  private:
};

template <typename SrcT, typename DstT = SrcT> class TestNppSrcDstBase : public TestNppBase
{
  public:
    TestNppSrcDstBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
    }
    virtual ~TestNppSrcDstBase() = default;

    TestNppSrcDstBase(const TestNppSrcDstBase &)     = default;
    TestNppSrcDstBase(TestNppSrcDstBase &&) noexcept = default;

    TestNppSrcDstBase &operator=(const TestNppSrcDstBase &)     = default;
    TestNppSrcDstBase &operator=(TestNppSrcDstBase &&) noexcept = default;

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
        dst.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
        dst.SetRoi(aBorder);
    }

    void Init() override
    {
    }

    SrcT &GetSrc1()
    {
        return src1;
    }

    DstT &GetDst()
    {
        return dst;
    }

  protected:
    SrcT src1;
    DstT dst;

  private:
};

template <typename Src1T, typename Src2T = Src1T, typename DstT = Src1T> class TestNppSrcSrcDstBase : public TestNppBase
{
  public:
    TestNppSrcSrcDstBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight), dst(aWidth, aHeight)
    {
    }
    virtual ~TestNppSrcSrcDstBase() = default;

    TestNppSrcSrcDstBase(const TestNppSrcSrcDstBase &)     = default;
    TestNppSrcSrcDstBase(TestNppSrcSrcDstBase &&) noexcept = default;

    TestNppSrcSrcDstBase &operator=(const TestNppSrcSrcDstBase &)     = default;
    TestNppSrcSrcDstBase &operator=(TestNppSrcSrcDstBase &&) noexcept = default;

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
        src2.SetRoi(aRoi);
        dst.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
        src2.SetRoi(aBorder);
        dst.SetRoi(aBorder);
    }

    void Init() override
    {
    }

    Src1T &GetSrc1()
    {
        return src1;
    }

    Src2T &GetSrc2()
    {
        return src2;
    }

    DstT &GetDst()
    {
        return dst;
    }

  protected:
    Src1T src1;
    Src2T src2;
    DstT dst;

  private:
};

template <typename SrcT, typename DstT, size_t dstSize> class TestNppSrcReductionBase : public TestNppBase
{
  public:
    TestNppSrcReductionBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
    }
    virtual ~TestNppSrcReductionBase() = default;

    TestNppSrcReductionBase(const TestNppSrcReductionBase &)     = default;
    TestNppSrcReductionBase(TestNppSrcReductionBase &&) noexcept = default;

    TestNppSrcReductionBase &operator=(const TestNppSrcReductionBase &)     = default;
    TestNppSrcReductionBase &operator=(TestNppSrcReductionBase &&) noexcept = default;

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
    }

    void Init() override
    {
    }

    SrcT &GetSrc1()
    {
        return src1;
    }

    DevVar<DstT> &GetDst()
    {
        return dst;
    }

  protected:
    SrcT src1;
    DevVar<byte> buffer{0};
    DevVar<DstT> dst{dstSize};

  private:
};

template <typename SrcT, typename DstT, size_t dstSize> class TestNppSrcSrcReductionBase : public TestNppBase
{
  public:
    TestNppSrcSrcReductionBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight)
    {
    }
    virtual ~TestNppSrcSrcReductionBase() = default;

    TestNppSrcSrcReductionBase(const TestNppSrcSrcReductionBase &)     = default;
    TestNppSrcSrcReductionBase(TestNppSrcSrcReductionBase &&) noexcept = default;

    TestNppSrcSrcReductionBase &operator=(const TestNppSrcSrcReductionBase &)     = default;
    TestNppSrcSrcReductionBase &operator=(TestNppSrcSrcReductionBase &&) noexcept = default;

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
        src2.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
        src2.SetRoi(aBorder);
    }

    void Init() override
    {
    }

    SrcT &GetSrc1()
    {
        return src1;
    }

    SrcT &GetSrc2()
    {
        return src2;
    }

    DevVar<DstT> &GetDst()
    {
        return dst;
    }

  protected:
    SrcT src1;
    SrcT src2;
    DevVar<byte> buffer{0};
    DevVar<DstT> dst{dstSize};

  private:
};
} // namespace opp