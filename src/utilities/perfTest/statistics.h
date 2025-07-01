#pragma once
#include "testMppBase.h"
#include "testNppBase.h"
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/filterArea.h>
#include <common/vector_typetraits.h>

namespace mpp
{
template <typename mppT, typename nppT> class StatisticsSrcBase
{
  public:
    StatisticsSrcBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : mpp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~StatisticsSrcBase() = default;

    StatisticsSrcBase(const StatisticsSrcBase &)     = default;
    StatisticsSrcBase(StatisticsSrcBase &&) noexcept = default;

    StatisticsSrcBase &operator=(const StatisticsSrcBase &)     = default;
    StatisticsSrcBase &operator=(StatisticsSrcBase &&) noexcept = default;

    virtual std::string GetName() = 0;
    virtual std::string GetType() = 0;
    virtual int GetOrder1()       = 0;
    virtual int GetOrder2()       = 0;
    virtual int GetOrder3()       = 0;

    template <typename ImgT> void Init(const ImgT &aCpuSrc1)
    {
        mpp.Init();
        npp.Init();

        aCpuSrc1 >> mpp.GetSrc1();
        aCpuSrc1 >> npp.GetSrc1();
    }

    void Run(const Roi &aRoi)
    {
        mpp.SetRoi(aRoi);
        npp.SetRoi(aRoi);
        mpp.AllocBuffer(); // might change with roi...
        npp.AllocBuffer(); // might change with roi...

        mpp.WarmUp();
        rt_mpp = mpp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();
    }

    template <PixelType mppT2, PixelType nppT2> TestResult GetResult()
    {
        std::cout << GetName() << " for " << GetType() << std::endl;

        nppT2 resNpp;
        mppT2 resMpp;

        npp.GetDst() >> resNpp.data();
        mpp.GetDst() >> resMpp.data();

        std::cout << "MPP:" << std::endl;
        std::cout << rt_mpp << std::endl;
        std::cout << "NPP:" << std::endl;
        std::cout << rt_npp << std::endl;

        const float ratio = rt_npp.Mean / rt_mpp.Mean;
        if (ratio >= 1.0f)
        {
            std::cout << "MPP is " << (rt_npp.Mean - rt_mpp.Mean) / static_cast<float>(repeats) << " msec or "
                      << ratio * 100.0f - 100.0f << "% faster!" << std::endl;
        }
        else
        {
            std::cout << "MPP is " << (rt_mpp.Mean - rt_npp.Mean) / static_cast<float>(repeats) << " msec or "
                      << 1.0f / ratio * 100.0f - 100.0f << "% slower..." << std::endl;
        }

        std::cout << "Result NPP: " << resNpp << " MPP: " << resMpp << std::endl;
        std::cout << std::endl
                  << "--------------------------------------------------------------------------------" << std::endl;

        TestResult res;

        res.Name                   = GetName() + " for " + GetType();
        res.TotalMPP               = rt_mpp.Total;
        res.TotalNPP               = rt_npp.Total;
        res.MeanMPP                = rt_mpp.Mean;
        res.MeanNPP                = rt_npp.Mean;
        res.StdMPP                 = rt_mpp.Std;
        res.StdNPP                 = rt_npp.Std;
        res.MinMPP                 = rt_mpp.Min;
        res.MinNPP                 = rt_npp.Min;
        res.MaxMPP                 = rt_mpp.Max;
        res.MaxNPP                 = rt_npp.Max;
        res.AbsoluteDifferenceMSec = (rt_npp.Mean - rt_mpp.Mean) / static_cast<float>(repeats);
        if (ratio >= 1.0f)
        {
            res.RelativeDifference = ratio * 100.0f - 100.0f;
        }
        else
        {
            res.RelativeDifference = -1.0f / ratio * 100.0f + 100.0f;
        }
        res.Order1 = GetOrder1();
        res.Order2 = GetOrder2();
        res.Order3 = GetOrder3();
        return res;
    }

  protected:
    mppT mpp;
    nppT npp;

    Runtime rt_mpp;
    Runtime rt_npp;

    size_t repeats;
};

template <typename mppT, typename nppT> class StatisticsSrcSrcBase
{
  public:
    StatisticsSrcSrcBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : mpp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~StatisticsSrcSrcBase() = default;

    StatisticsSrcSrcBase(const StatisticsSrcSrcBase &)     = default;
    StatisticsSrcSrcBase(StatisticsSrcSrcBase &&) noexcept = default;

    StatisticsSrcSrcBase &operator=(const StatisticsSrcSrcBase &)     = default;
    StatisticsSrcSrcBase &operator=(StatisticsSrcSrcBase &&) noexcept = default;

    virtual std::string GetName() = 0;
    virtual std::string GetType() = 0;
    virtual int GetOrder1()       = 0;
    virtual int GetOrder2()       = 0;
    virtual int GetOrder3()       = 0;

    template <typename ImgT> void Init(const ImgT &aCpuSrc1, const ImgT &aCpuSrc2)
    {
        mpp.Init();
        npp.Init();

        aCpuSrc1 >> mpp.GetSrc1();
        aCpuSrc1 >> npp.GetSrc1();

        aCpuSrc2 >> mpp.GetSrc2();
        aCpuSrc2 >> npp.GetSrc2();
    }

    void Run(const Roi &aRoi)
    {
        mpp.SetRoi(aRoi);
        npp.SetRoi(aRoi);
        mpp.AllocBuffer(); // might change with roi...
        npp.AllocBuffer(); // might change with roi...

        mpp.WarmUp();
        rt_mpp = mpp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();
    }

    template <PixelType mppT2, PixelType nppT2> TestResult GetResult()
    {
        std::cout << GetName() << " for " << GetType() << std::endl;

        nppT2 resNpp;
        mppT2 resMpp;

        npp.GetDst() >> resNpp.data();
        mpp.GetDst() >> resMpp.data();

        std::cout << "MPP:" << std::endl;
        std::cout << rt_mpp << std::endl;
        std::cout << "NPP:" << std::endl;
        std::cout << rt_npp << std::endl;

        const float ratio = rt_npp.Mean / rt_mpp.Mean;
        if (ratio >= 1.0f)
        {
            std::cout << "MPP is " << (rt_npp.Mean - rt_mpp.Mean) / static_cast<float>(repeats) << " msec or "
                      << ratio * 100.0f - 100.0f << "% faster!" << std::endl;
        }
        else
        {
            std::cout << "MPP is " << (rt_mpp.Mean - rt_npp.Mean) / static_cast<float>(repeats) << " msec or "
                      << 1.0f / ratio * 100.0f - 100.0f << "% slower..." << std::endl;
        }

        std::cout << "Result NPP: " << resNpp << " MPP: " << resMpp << std::endl;
        std::cout << std::endl
                  << "--------------------------------------------------------------------------------" << std::endl;

        TestResult res;

        res.Name                   = GetName() + " for " + GetType();
        res.TotalMPP               = rt_mpp.Total;
        res.TotalNPP               = rt_npp.Total;
        res.MeanMPP                = rt_mpp.Mean;
        res.MeanNPP                = rt_npp.Mean;
        res.StdMPP                 = rt_mpp.Std;
        res.StdNPP                 = rt_npp.Std;
        res.MinMPP                 = rt_mpp.Min;
        res.MinNPP                 = rt_npp.Min;
        res.MaxMPP                 = rt_mpp.Max;
        res.MaxNPP                 = rt_npp.Max;
        res.AbsoluteDifferenceMSec = (rt_npp.Mean - rt_mpp.Mean) / static_cast<float>(repeats);
        if (ratio >= 1.0f)
        {
            res.RelativeDifference = ratio * 100.0f - 100.0f;
        }
        else
        {
            res.RelativeDifference = -1.0f / ratio * 100.0f + 100.0f;
        }
        res.Order1 = GetOrder1();
        res.Order2 = GetOrder2();
        res.Order3 = GetOrder3();
        return res;
    }

  protected:
    mppT mpp;
    nppT npp;

    Runtime rt_mpp;
    Runtime rt_npp;

    size_t repeats;
};

template <PixelType SrcT, typename DstT> class QualityIndexMpp : public TestMppSrcSrcReductionBase<SrcT, DstT>
{
  public:
    QualityIndexMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.QualityIndexBufferSize(this->ctx));
    }

    void RunOnce() override
    {
        this->src1.QualityIndex(this->src2, this->dst, this->buffer, this->ctx);
    }

  private:
};

template <typename SrcT, typename DstT, size_t resSize>
class QualityIndexNpp : public TestNppSrcSrcReductionBase<SrcT, DstT, resSize>
{
  public:
    QualityIndexNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcSrcReductionBase<SrcT, DstT, resSize>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.QualityIndexGetBufferHostSize(this->ctx));
    }

    void RunOnce() override
    {
        this->src1.QualityIndex(this->src2, this->src1.NppiSizeRoi(), this->dst, this->buffer, this->ctx);
    }

  private:
};

template <PixelType SrcT, typename DstT, typename SrcT2, typename DstT2, size_t resSize>
class QualityIndexTest
    : public StatisticsSrcSrcBase<QualityIndexMpp<SrcT, DstT>, QualityIndexNpp<SrcT2, DstT2, resSize>>
{
  public:
    QualityIndexTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcSrcBase<QualityIndexMpp<SrcT, DstT>, QualityIndexNpp<SrcT2, DstT2, resSize>>(
              aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "QualityIndex";
    }

    std::string GetType() override
    {
        return pixel_type_name<SrcT>::value;
    }
    int GetOrder1() override
    {
        return 600 + 0;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<SrcT>)
        {
            return channel_count_v<SrcT> + 64 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else if constexpr (RealSignedVector<SrcT>)
        {
            return channel_count_v<SrcT> + 32 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else
        {
            return channel_count_v<SrcT> + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
    }
};

template <PixelType SrcT, typename DstT> class MaxMpp : public TestMppSrcReductionBase<SrcT, DstT>
{
  public:
    MaxMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MaxBufferSize(this->ctx));
    }

    void RunOnce() override
    {
        if constexpr (vector_size_v<SrcT> == 1)
        {
            this->src1.Max(this->dst, this->buffer, this->ctx);
        }
        else
        {
            auto null = DevVarView<remove_vector_t<DstT>>::Null();
            this->src1.Max(this->dst, null, this->buffer, this->ctx);
        }
    }

  private:
};

template <typename SrcT, typename DstT, size_t resSize>
class MaxNpp : public TestNppSrcReductionBase<SrcT, DstT, resSize>
{
  public:
    MaxNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcReductionBase<SrcT, DstT, resSize>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MaxGetBufferHostSize(this->ctx));
    }

    void RunOnce() override
    {
        this->src1.Max(this->buffer, this->dst, this->ctx);
    }

  private:
};

template <PixelType SrcT, typename DstT, typename SrcT2, typename DstT2, size_t resSize>
class MaxTest : public StatisticsSrcBase<MaxMpp<SrcT, DstT>, MaxNpp<SrcT2, DstT2, resSize>>
{
  public:
    MaxTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcBase<MaxMpp<SrcT, DstT>, MaxNpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "Max";
    }

    std::string GetType() override
    {
        return pixel_type_name<SrcT>::value;
    }
    int GetOrder1() override
    {
        return 600 + 1;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<SrcT>)
        {
            return channel_count_v<SrcT> + 64 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else if constexpr (RealSignedVector<SrcT>)
        {
            return channel_count_v<SrcT> + 32 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else
        {
            return channel_count_v<SrcT> + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
    }
};

template <PixelType SrcT, typename DstT> class MeanMpp : public TestMppSrcReductionBase<SrcT, DstT>
{
  public:
    MeanMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MeanBufferSize(this->ctx));
    }

    void RunOnce() override
    {
        if constexpr (vector_size_v<SrcT> == 1)
        {
            this->src1.Mean(this->dst, this->buffer, this->ctx);
        }
        else
        {
            auto null = DevVarView<remove_vector_t<DstT>>::Null();
            this->src1.Mean(this->dst, null, this->buffer, this->ctx);
        }
    }

  private:
};

template <typename SrcT, typename DstT, size_t resSize>
class MeanNpp : public TestNppSrcReductionBase<SrcT, DstT, resSize>
{
  public:
    MeanNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcReductionBase<SrcT, DstT, resSize>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MeanGetBufferHostSize(this->ctx));
    }

    void RunOnce() override
    {
        this->src1.Mean(this->buffer, this->dst, this->ctx);
    }

  private:
};

template <PixelType SrcT, typename DstT, typename SrcT2, typename DstT2, size_t resSize>
class MeanTest : public StatisticsSrcBase<MeanMpp<SrcT, DstT>, MeanNpp<SrcT2, DstT2, resSize>>
{
  public:
    MeanTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcBase<MeanMpp<SrcT, DstT>, MeanNpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "Mean";
    }

    std::string GetType() override
    {
        return pixel_type_name<SrcT>::value;
    }
    int GetOrder1() override
    {
        return 600 + 2;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<SrcT>)
        {
            return channel_count_v<SrcT> + 64 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else if constexpr (RealSignedVector<SrcT>)
        {
            return channel_count_v<SrcT> + 32 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else
        {
            return channel_count_v<SrcT> + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
    }
};

template <PixelType SrcT, typename DstT> class MeanStdMpp : public TestMppSrcReductionBase<SrcT, DstT>
{
  public:
    MeanStdMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MeanStdBufferSize(this->ctx));
    }

    void RunOnce() override
    {
        if constexpr (vector_size_v<SrcT> == 1)
        {
            this->src1.MeanStd(mean, this->dst, this->buffer, this->ctx);
        }
        else
        {
            auto null = DevVarView<remove_vector_t<DstT>>::Null();
            this->src1.MeanStd(mean, this->dst, null, null, this->buffer, this->ctx);
        }
    }

  private:
    DevVar<DstT> mean{1};
};

template <typename SrcT, typename DstT, size_t resSize>
class MeanStdNpp : public TestNppSrcReductionBase<SrcT, DstT, resSize>
{
  public:
    MeanStdNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcReductionBase<SrcT, DstT, resSize>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MeanStdDevGetBufferHostSize(this->ctx));
    }

    void RunOnce() override
    {
        if constexpr (resSize == 1)
        {
            this->src1.Mean_StdDev(this->buffer, mean, this->dst, this->ctx);
        }
        else
        {
            // there is no 4 channel version of Mean_StdDev...
            auto DstChannel1  = DevVarView<DstT>(this->dst.Pointer() + 1, sizeof(DstT));
            auto DstChannel2  = DevVarView<DstT>(this->dst.Pointer() + 2, sizeof(DstT));
            auto MeanChannel1 = DevVarView<DstT>(mean.Pointer() + 1, sizeof(DstT));
            auto MeanChannel2 = DevVarView<DstT>(mean.Pointer() + 2, sizeof(DstT));
            this->src1.Mean_StdDev(1, this->buffer, mean, this->dst, this->ctx);
            this->src1.Mean_StdDev(2, this->buffer, MeanChannel1, DstChannel1, this->ctx);
            this->src1.Mean_StdDev(3, this->buffer, MeanChannel2, DstChannel2, this->ctx);
        }
    }

  private:
    DevVar<DstT> mean{resSize};
};

template <PixelType SrcT, typename DstT, typename SrcT2, typename DstT2, size_t resSize>
class MeanStdTest : public StatisticsSrcBase<MeanStdMpp<SrcT, DstT>, MeanStdNpp<SrcT2, DstT2, resSize>>
{
  public:
    MeanStdTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcBase<MeanStdMpp<SrcT, DstT>, MeanStdNpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth,
                                                                                       aHeight)
    {
    }

    std::string GetName() override
    {
        return "MeanStd";
    }

    std::string GetType() override
    {
        return pixel_type_name<SrcT>::value;
    }
    int GetOrder1() override
    {
        return 600 + 3;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<SrcT>)
        {
            return channel_count_v<SrcT> + 64 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else if constexpr (RealSignedVector<SrcT>)
        {
            return channel_count_v<SrcT> + 32 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else
        {
            return channel_count_v<SrcT> + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
    }
};

template <PixelType SrcT, typename DstT> class MSEMpp : public TestMppSrcSrcReductionBase<SrcT, DstT>
{
  public:
    MSEMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MSEBufferSize(this->ctx));
    }

    void RunOnce() override
    {
        if constexpr (vector_size_v<SrcT> == 1)
        {
            this->src1.MSE(this->src2, this->dst, this->buffer, this->ctx);
        }
        else
        {
            auto null = DevVarView<remove_vector_t<DstT>>::Null();
            this->src1.MSE(this->src2, this->dst, null, this->buffer, this->ctx);
        }
    }

  private:
};

template <typename SrcT, typename DstT, size_t resSize>
class MSENpp : public TestNppSrcSrcReductionBase<SrcT, DstT, resSize>
{
  public:
    MSENpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcSrcReductionBase<SrcT, DstT, resSize>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void AllocBuffer()
    {
        this->buffer = DevVar<byte>(this->src1.MSEGetBufferHostSize(this->ctx));
    }

    void RunOnce() override
    {
        this->src1.MSE(this->src2, this->dst, this->buffer, this->ctx);
    }

  private:
};

template <PixelType SrcT, typename DstT, typename SrcT2, typename DstT2, size_t resSize>
class MSETest : public StatisticsSrcSrcBase<MSEMpp<SrcT, DstT>, MSENpp<SrcT2, DstT2, resSize>>
{
  public:
    MSETest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcSrcBase<MSEMpp<SrcT, DstT>, MSENpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth,
                                                                                  aHeight)
    {
    }

    std::string GetName() override
    {
        return "MSE";
    }

    std::string GetType() override
    {
        return pixel_type_name<SrcT>::value;
    }
    int GetOrder1() override
    {
        return 600 + 4;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<SrcT>)
        {
            return channel_count_v<SrcT> + 64 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else if constexpr (RealSignedVector<SrcT>)
        {
            return channel_count_v<SrcT> + 32 + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
        else
        {
            return channel_count_v<SrcT> + sizeof(pixel_basetype_t<SrcT>) * 128;
        }
    }
};
} // namespace mpp