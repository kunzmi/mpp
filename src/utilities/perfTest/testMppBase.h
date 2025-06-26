#pragma once
#include "testBase.h"
#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/pixelTypes.h>

namespace mpp
{
using namespace mpp::cuda;
using namespace mpp::image;
using namespace mpp::image::cuda;

class TestMppBase : public TestBase
{
  public:
    TestMppBase(size_t aIterations, size_t aRepeats) : TestBase(aIterations, aRepeats)
    {
        ctx = StreamCtxSingleton::Get();
    }
    virtual ~TestMppBase() = default;

    TestMppBase(const TestMppBase &)     = default;
    TestMppBase(TestMppBase &&) noexcept = default;

    TestMppBase &operator=(const TestMppBase &)     = default;
    TestMppBase &operator=(TestMppBase &&) noexcept = default;

  protected:
    StreamCtx ctx;

  private:
};

template <PixelType SrcT, PixelType DstT = SrcT> class TestMppSrcDstBase : public TestMppBase
{
  public:
    TestMppSrcDstBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
    }
    virtual ~TestMppSrcDstBase() = default;

    TestMppSrcDstBase(const TestMppSrcDstBase &)     = default;
    TestMppSrcDstBase(TestMppSrcDstBase &&) noexcept = default;

    TestMppSrcDstBase &operator=(const TestMppSrcDstBase &)     = default;
    TestMppSrcDstBase &operator=(TestMppSrcDstBase &&) noexcept = default;

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
class TestMppSrcSrcDstBase : public TestMppBase
{
  public:
    TestMppSrcSrcDstBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight), dst(aWidth, aHeight)
    {
    }
    virtual ~TestMppSrcSrcDstBase() = default;

    TestMppSrcSrcDstBase(const TestMppSrcSrcDstBase &)     = default;
    TestMppSrcSrcDstBase(TestMppSrcSrcDstBase &&) noexcept = default;

    TestMppSrcSrcDstBase &operator=(const TestMppSrcSrcDstBase &)     = default;
    TestMppSrcSrcDstBase &operator=(TestMppSrcSrcDstBase &&) noexcept = default;

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

template <PixelType SrcT, typename DstT> class TestMppSrcReductionBase : public TestMppBase
{
  public:
    TestMppSrcReductionBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
    }
    virtual ~TestMppSrcReductionBase() = default;

    TestMppSrcReductionBase(const TestMppSrcReductionBase &)     = default;
    TestMppSrcReductionBase(TestMppSrcReductionBase &&) noexcept = default;

    TestMppSrcReductionBase &operator=(const TestMppSrcReductionBase &)     = default;
    TestMppSrcReductionBase &operator=(TestMppSrcReductionBase &&) noexcept = default;

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

template <PixelType SrcT, typename DstT> class TestMppSrcSrcReductionBase : public TestMppBase
{
  public:
    TestMppSrcSrcReductionBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight)
    {
    }
    virtual ~TestMppSrcSrcReductionBase() = default;

    TestMppSrcSrcReductionBase(const TestMppSrcSrcReductionBase &)     = default;
    TestMppSrcSrcReductionBase(TestMppSrcSrcReductionBase &&) noexcept = default;

    TestMppSrcSrcReductionBase &operator=(const TestMppSrcSrcReductionBase &)     = default;
    TestMppSrcSrcReductionBase &operator=(TestMppSrcSrcReductionBase &&) noexcept = default;

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
} // namespace mpp