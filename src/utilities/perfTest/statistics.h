#pragma once
#include "testNppBase.h"
#include "testOppBase.h"
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/filterArea.h>
#include <common/vector_typetraits.h>

namespace opp
{
template <typename oppT, typename nppT> class StatisticsSrcBase
{
  public:
    StatisticsSrcBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : opp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~StatisticsSrcBase() = default;

    StatisticsSrcBase(const StatisticsSrcBase &)     = default;
    StatisticsSrcBase(StatisticsSrcBase &&) noexcept = default;

    StatisticsSrcBase &operator=(const StatisticsSrcBase &)     = default;
    StatisticsSrcBase &operator=(StatisticsSrcBase &&) noexcept = default;

    virtual std::string GetName() = 0;
    virtual std::string GetType() = 0;

    template <typename ImgT> void Init(const ImgT &aCpuSrc1)
    {
        opp.Init();
        npp.Init();

        aCpuSrc1 >> opp.GetSrc1();
        aCpuSrc1 >> npp.GetSrc1();
    }

    void Run(const Roi &aRoi)
    {
        opp.SetRoi(aRoi);
        npp.SetRoi(aRoi);
        opp.AllocBuffer(); // might change with roi...
        npp.AllocBuffer(); // might change with roi...

        opp.WarmUp();
        rt_opp = opp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();
    }

    template <PixelType oppT2, PixelType nppT2> TestResult GetResult()
    {
        std::cout << GetName() << " for " << GetType() << std::endl;

        nppT2 resNpp;
        oppT2 resOpp;

        npp.GetDst() >> resNpp.data();
        opp.GetDst() >> resOpp.data();

        std::cout << "OPP:" << std::endl;
        std::cout << rt_opp << std::endl;
        std::cout << "NPP:" << std::endl;
        std::cout << rt_npp << std::endl;

        const float ratio = rt_npp.Mean / rt_opp.Mean;
        if (ratio >= 1.0f)
        {
            std::cout << "OPP is " << (rt_npp.Mean - rt_opp.Mean) / static_cast<float>(repeats) << " msec or "
                      << ratio * 100.0f - 100.0f << "% faster!" << std::endl;
        }
        else
        {
            std::cout << "OPP is " << (rt_opp.Mean - rt_npp.Mean) / static_cast<float>(repeats) << " msec or "
                      << 1.0f / ratio * 100.0f - 100.0f << "% slower..." << std::endl;
        }

        std::cout << "Result NPP: " << resNpp << " OPP: " << resOpp << std::endl;
        std::cout << std::endl
                  << "--------------------------------------------------------------------------------" << std::endl;

        TestResult res;

        res.Name                   = GetName() + " for " + GetType();
        res.TotalOPP               = rt_opp.Total;
        res.TotalNPP               = rt_npp.Total;
        res.MeanOPP                = rt_opp.Mean;
        res.MeanNPP                = rt_npp.Mean;
        res.StdOPP                 = rt_opp.Std;
        res.StdNPP                 = rt_npp.Std;
        res.MinOPP                 = rt_opp.Min;
        res.MinNPP                 = rt_npp.Min;
        res.MaxOPP                 = rt_opp.Max;
        res.MaxNPP                 = rt_npp.Max;
        res.AbsoluteDifferenceMSec = (rt_npp.Mean - rt_opp.Mean) / static_cast<float>(repeats);
        if (ratio >= 1.0f)
        {
            res.RelativeDifference = ratio * 100.0f - 100.0f;
        }
        else
        {
            res.RelativeDifference = -1.0f / ratio * 100.0f + 100.0f;
        }
        return res;
    }

  protected:
    oppT opp;
    nppT npp;

    Runtime rt_opp;
    Runtime rt_npp;

    size_t repeats;
};

template <typename oppT, typename nppT> class StatisticsSrcSrcBase
{
  public:
    StatisticsSrcSrcBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : opp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~StatisticsSrcSrcBase() = default;

    StatisticsSrcSrcBase(const StatisticsSrcSrcBase &)     = default;
    StatisticsSrcSrcBase(StatisticsSrcSrcBase &&) noexcept = default;

    StatisticsSrcSrcBase &operator=(const StatisticsSrcSrcBase &)     = default;
    StatisticsSrcSrcBase &operator=(StatisticsSrcSrcBase &&) noexcept = default;

    virtual std::string GetName() = 0;
    virtual std::string GetType() = 0;

    template <typename ImgT> void Init(const ImgT &aCpuSrc1, const ImgT &aCpuSrc2)
    {
        opp.Init();
        npp.Init();

        aCpuSrc1 >> opp.GetSrc1();
        aCpuSrc1 >> npp.GetSrc1();

        aCpuSrc2 >> opp.GetSrc2();
        aCpuSrc2 >> npp.GetSrc2();
    }

    void Run(const Roi &aRoi)
    {
        opp.SetRoi(aRoi);
        npp.SetRoi(aRoi);
        opp.AllocBuffer(); // might change with roi...
        npp.AllocBuffer(); // might change with roi...

        opp.WarmUp();
        rt_opp = opp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();
    }

    template <PixelType oppT2, PixelType nppT2> TestResult GetResult()
    {
        std::cout << GetName() << " for " << GetType() << std::endl;

        nppT2 resNpp;
        oppT2 resOpp;

        npp.GetDst() >> resNpp.data();
        opp.GetDst() >> resOpp.data();

        std::cout << "OPP:" << std::endl;
        std::cout << rt_opp << std::endl;
        std::cout << "NPP:" << std::endl;
        std::cout << rt_npp << std::endl;

        const float ratio = rt_npp.Mean / rt_opp.Mean;
        if (ratio >= 1.0f)
        {
            std::cout << "OPP is " << (rt_npp.Mean - rt_opp.Mean) / static_cast<float>(repeats) << " msec or "
                      << ratio * 100.0f - 100.0f << "% faster!" << std::endl;
        }
        else
        {
            std::cout << "OPP is " << (rt_opp.Mean - rt_npp.Mean) / static_cast<float>(repeats) << " msec or "
                      << 1.0f / ratio * 100.0f - 100.0f << "% slower..." << std::endl;
        }

        std::cout << "Result NPP: " << resNpp << " OPP: " << resOpp << std::endl;
        std::cout << std::endl
                  << "--------------------------------------------------------------------------------" << std::endl;

        TestResult res;

        res.Name                   = GetName() + " for " + GetType();
        res.TotalOPP               = rt_opp.Total;
        res.TotalNPP               = rt_npp.Total;
        res.MeanOPP                = rt_opp.Mean;
        res.MeanNPP                = rt_npp.Mean;
        res.StdOPP                 = rt_opp.Std;
        res.StdNPP                 = rt_npp.Std;
        res.MinOPP                 = rt_opp.Min;
        res.MinNPP                 = rt_npp.Min;
        res.MaxOPP                 = rt_opp.Max;
        res.MaxNPP                 = rt_npp.Max;
        res.AbsoluteDifferenceMSec = (rt_npp.Mean - rt_opp.Mean) / static_cast<float>(repeats);
        if (ratio >= 1.0f)
        {
            res.RelativeDifference = ratio * 100.0f - 100.0f;
        }
        else
        {
            res.RelativeDifference = -1.0f / ratio * 100.0f + 100.0f;
        }
        return res;
    }

  protected:
    oppT opp;
    nppT npp;

    Runtime rt_opp;
    Runtime rt_npp;

    size_t repeats;
};

template <PixelType SrcT, typename DstT> class QualityIndexOpp : public TestOppSrcSrcReductionBase<SrcT, DstT>
{
  public:
    QualityIndexOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
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
    : public StatisticsSrcSrcBase<QualityIndexOpp<SrcT, DstT>, QualityIndexNpp<SrcT2, DstT2, resSize>>
{
  public:
    QualityIndexTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcSrcBase<QualityIndexOpp<SrcT, DstT>, QualityIndexNpp<SrcT2, DstT2, resSize>>(
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
};

template <PixelType SrcT, typename DstT> class MaxOpp : public TestOppSrcReductionBase<SrcT, DstT>
{
  public:
    MaxOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
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
class MaxTest : public StatisticsSrcBase<MaxOpp<SrcT, DstT>, MaxNpp<SrcT2, DstT2, resSize>>
{
  public:
    MaxTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcBase<MaxOpp<SrcT, DstT>, MaxNpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth, aHeight)
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
};

template <PixelType SrcT, typename DstT> class MeanOpp : public TestOppSrcReductionBase<SrcT, DstT>
{
  public:
    MeanOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
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
class MeanTest : public StatisticsSrcBase<MeanOpp<SrcT, DstT>, MeanNpp<SrcT2, DstT2, resSize>>
{
  public:
    MeanTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcBase<MeanOpp<SrcT, DstT>, MeanNpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth, aHeight)
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
};

template <PixelType SrcT, typename DstT> class MeanStdOpp : public TestOppSrcReductionBase<SrcT, DstT>
{
  public:
    MeanStdOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
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
class MeanStdTest : public StatisticsSrcBase<MeanStdOpp<SrcT, DstT>, MeanStdNpp<SrcT2, DstT2, resSize>>
{
  public:
    MeanStdTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcBase<MeanStdOpp<SrcT, DstT>, MeanStdNpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth,
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
};

template <PixelType SrcT, typename DstT> class MSEOpp : public TestOppSrcSrcReductionBase<SrcT, DstT>
{
  public:
    MSEOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcSrcReductionBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
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
class MSETest : public StatisticsSrcSrcBase<MSEOpp<SrcT, DstT>, MSENpp<SrcT2, DstT2, resSize>>
{
  public:
    MSETest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : StatisticsSrcSrcBase<MSEOpp<SrcT, DstT>, MSENpp<SrcT2, DstT2, resSize>>(aIterations, aRepeats, aWidth,
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
};
} // namespace opp