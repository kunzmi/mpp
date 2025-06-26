#pragma once
#include "testBase.h"
#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/pixelTypes.h>

namespace opp
{
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;

class TestOppBase : public TestBase
{
  public:
    TestOppBase(size_t aIterations, size_t aRepeats) : TestBase(aIterations, aRepeats)
    {
        ctx = StreamCtxSingleton::Get();
    }
    virtual ~TestOppBase() = default;

    TestOppBase(const TestOppBase &)     = default;
    TestOppBase(TestOppBase &&) noexcept = default;

    TestOppBase &operator=(const TestOppBase &)     = default;
    TestOppBase &operator=(TestOppBase &&) noexcept = default;

  protected:
    StreamCtx ctx;

  private:
};

template <PixelType SrcT, PixelType DstT = SrcT> class TestOppSrcDstBase : public TestOppBase
{
  public:
    TestOppSrcDstBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
    }
    virtual ~TestOppSrcDstBase() = default;

    TestOppSrcDstBase(const TestOppSrcDstBase &)     = default;
    TestOppSrcDstBase(TestOppSrcDstBase &&) noexcept = default;

    TestOppSrcDstBase &operator=(const TestOppSrcDstBase &)     = default;
    TestOppSrcDstBase &operator=(TestOppSrcDstBase &&) noexcept = default;

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

    Image<SrcT> &GetSrc1()
    {
        return src1;
    }

    Image<DstT> &GetDst()
    {
        return dst;
    }

  protected:
    Image<SrcT> src1;
    Image<DstT> dst;

  private:
};

template <PixelType Src1T, PixelType Src2T = Src1T, PixelType DstT = Src1T>
class TestOppSrcSrcDstBase : public TestOppBase
{
  public:
    TestOppSrcSrcDstBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight), dst(aWidth, aHeight)
    {
    }
    virtual ~TestOppSrcSrcDstBase() = default;

    TestOppSrcSrcDstBase(const TestOppSrcSrcDstBase &)     = default;
    TestOppSrcSrcDstBase(TestOppSrcSrcDstBase &&) noexcept = default;

    TestOppSrcSrcDstBase &operator=(const TestOppSrcSrcDstBase &)     = default;
    TestOppSrcSrcDstBase &operator=(TestOppSrcSrcDstBase &&) noexcept = default;

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

    Image<Src1T> &GetSrc1()
    {
        return src1;
    }

    Image<Src2T> &GetSrc2()
    {
        return src2;
    }

    Image<DstT> &GetDst()
    {
        return dst;
    }

  protected:
    Image<Src1T> src1;
    Image<Src2T> src2;
    Image<DstT> dst;

  private:
};

template <PixelType SrcT, typename DstT> class TestOppSrcReductionBase : public TestOppBase
{
  public:
    TestOppSrcReductionBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
    }
    virtual ~TestOppSrcReductionBase() = default;

    TestOppSrcReductionBase(const TestOppSrcReductionBase &)     = default;
    TestOppSrcReductionBase(TestOppSrcReductionBase &&) noexcept = default;

    TestOppSrcReductionBase &operator=(const TestOppSrcReductionBase &)     = default;
    TestOppSrcReductionBase &operator=(TestOppSrcReductionBase &&) noexcept = default;

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

    Image<SrcT> &GetSrc1()
    {
        return src1;
    }

    DevVar<DstT> &GetDst()
    {
        return dst;
    }

  protected:
    Image<SrcT> src1;
    DevVar<byte> buffer{0};
    DevVar<DstT> dst{1};

  private:
};

template <PixelType SrcT, typename DstT> class TestOppSrcSrcReductionBase : public TestOppBase
{
  public:
    TestOppSrcSrcReductionBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight)
    {
    }
    virtual ~TestOppSrcSrcReductionBase() = default;

    TestOppSrcSrcReductionBase(const TestOppSrcSrcReductionBase &)     = default;
    TestOppSrcSrcReductionBase(TestOppSrcSrcReductionBase &&) noexcept = default;

    TestOppSrcSrcReductionBase &operator=(const TestOppSrcSrcReductionBase &)     = default;
    TestOppSrcSrcReductionBase &operator=(TestOppSrcSrcReductionBase &&) noexcept = default;

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

    Image<SrcT> &GetSrc1()
    {
        return src1;
    }

    Image<SrcT> &GetSrc2()
    {
        return src2;
    }

    DevVar<DstT> &GetDst()
    {
        return dst;
    }

  protected:
    Image<SrcT> src1;
    Image<SrcT> src2;
    DevVar<byte> buffer{0};
    DevVar<DstT> dst{1};

  private:
};
} // namespace opp