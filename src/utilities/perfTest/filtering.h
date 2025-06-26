#pragma once
#include "testNppBase.h"
#include "testOppBase.h"
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/filterArea.h>
#include <common/vector_typetraits.h>

namespace opp
{
template <typename oppT, typename nppT> class FilteringBase
{
  public:
    FilteringBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : opp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~FilteringBase() = default;

    FilteringBase(const FilteringBase &)     = default;
    FilteringBase(FilteringBase &&) noexcept = default;

    FilteringBase &operator=(const FilteringBase &)     = default;
    FilteringBase &operator=(FilteringBase &&) noexcept = default;

    virtual std::string GetName() = 0;
    virtual std::string GetType() = 0;

    template <typename ImgT> void Init(const ImgT &aCpuSrc1)
    {
        opp.Init();
        npp.Init();

        aCpuSrc1 >> opp.GetSrc1();

        aCpuSrc1 >> npp.GetSrc1();
    }

    void Run(const Roi &aRoi, const FilterArea &aFilterArea)
    {
        opp.fa = aFilterArea;
        npp.fa = aFilterArea;
        opp.SetRoi(aRoi);
        npp.SetRoi(aRoi);

        opp.WarmUp();
        rt_opp = opp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();
    }

    template <typename ImgT, PixelType resT> TestResult GetResult()
    {
        std::stringstream ss;
        ss << GetName() << " for " << GetType() << " and filter size " << opp.fa.Size;

        std::cout << ss.str() << std::endl;

        ImgT resNpp(npp.GetDst().SizeAlloc());
        ImgT resOpp(opp.GetDst().SizeAlloc());

        resNpp << npp.GetDst();
        resOpp << opp.GetDst();

        resNpp.SetRoi(npp.GetDst().ROI());
        resOpp.SetRoi(opp.GetDst().ROI());

        double maxErrorScalar = 0;
        resT maxError{0};

        if constexpr (vector_size_v<resT> == 1)
        {
            resOpp.MaximumError(resNpp, maxError);
            maxErrorScalar = maxError.x;
        }
        else
        {
            resOpp.MaximumError(resNpp, maxError, maxErrorScalar);
        }
        double avgErrorScalar = 0;
        resT avgError{0};

        if constexpr (vector_size_v<resT> == 1)
        {
            resOpp.AverageError(resNpp, avgError);
            avgErrorScalar = avgError.x;
        }
        else
        {
            resOpp.AverageError(resNpp, avgError, avgErrorScalar);
        }

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

        std::cout << "MaxError: " << maxErrorScalar << " AvgError: " << avgErrorScalar << std::endl;
        std::cout << std::endl
                  << "--------------------------------------------------------------------------------" << std::endl;

        TestResult res;

        res.Name                   = ss.str();
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

template <PixelType T> class BoxFilterOpp : public TestOppSrcDstBase<T>
{
  public:
    BoxFilterOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.BoxFilter(this->dst, fa, BorderType::Replicate, this->ctx);
    }

    FilterArea fa{11};

  private:
};

template <typename T> class BoxFilterNpp : public TestNppSrcDstBase<T>
{
  public:
    BoxFilterNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.FilterBoxBorder(this->dst, {fa.Size.x, fa.Size.y}, {fa.Center.x, fa.Center.y}, NPP_BORDER_REPLICATE,
                                   this->ctx);
    }

    FilterArea fa{11};

  private:
};

template <PixelType T, typename T2> class BoxFilterTest : public FilteringBase<BoxFilterOpp<T>, BoxFilterNpp<T2>>
{
  public:
    BoxFilterTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : FilteringBase<BoxFilterOpp<T>, BoxFilterNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "BoxFilter";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};

template <PixelType SrcT, PixelType DstT> class ColumnWindowSumOpp : public TestOppSrcDstBase<SrcT, DstT>
{
  public:
    ColumnWindowSumOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.ColumnWindowSum(this->dst, 1.0f, fa.Size.y, fa.Center.y, BorderType::Replicate, this->ctx);
    }

    FilterArea fa{{1, 11}, {0, 5}};

  private:
};

template <typename SrcT, typename DstT> class ColumnWindowSumNpp : public TestNppSrcDstBase<SrcT, DstT>
{
  public:
    ColumnWindowSumNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.SumWindowColumnBorder(this->dst, fa.Size.y, fa.Center.y, NPP_BORDER_REPLICATE, this->ctx);
    }

    FilterArea fa{{1, 11}, {0, 5}};

  private:
};

template <PixelType SrcT, PixelType DstT, typename T2, typename DstT2>
class ColumnWindowSumTest : public FilteringBase<ColumnWindowSumOpp<SrcT, DstT>, ColumnWindowSumNpp<T2, DstT2>>
{
  public:
    ColumnWindowSumTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : FilteringBase<ColumnWindowSumOpp<SrcT, DstT>, ColumnWindowSumNpp<T2, DstT2>>(aIterations, aRepeats, aWidth,
                                                                                       aHeight)
    {
    }

    std::string GetName() override
    {
        return "ColumnWindowSum";
    }

    std::string GetType() override
    {
        return pixel_type_name<SrcT>::value;
    }
};

template <PixelType SrcT, PixelType DstT> class RowWindowSumOpp : public TestOppSrcDstBase<SrcT, DstT>
{
  public:
    RowWindowSumOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.RowWindowSum(this->dst, 1.0f, fa.Size.x, fa.Center.x, BorderType::Replicate, this->ctx);
    }

    FilterArea fa{{11, 1}, {5, 0}};

  private:
};

template <typename SrcT, typename DstT> class RowWindowSumNpp : public TestNppSrcDstBase<SrcT, DstT>
{
  public:
    RowWindowSumNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<SrcT, DstT>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.SumWindowRowBorder(this->dst, fa.Size.x, fa.Center.x, NPP_BORDER_REPLICATE, this->ctx);
    }

    FilterArea fa{{11, 1}, {5, 0}};

  private:
};

template <PixelType SrcT, PixelType DstT, typename T2, typename DstT2>
class RowWindowSumTest : public FilteringBase<RowWindowSumOpp<SrcT, DstT>, RowWindowSumNpp<T2, DstT2>>
{
  public:
    RowWindowSumTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : FilteringBase<RowWindowSumOpp<SrcT, DstT>, RowWindowSumNpp<T2, DstT2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "RowWindowSum";
    }

    std::string GetType() override
    {
        return pixel_type_name<SrcT>::value;
    }
};

template <PixelType T> class LowPassFilterOpp : public TestOppSrcDstBase<T>
{
  public:
    LowPassFilterOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        MaskSize ms = MaskSize::Mask_3x3;
        if (fa.Size == 5)
        {
            ms = MaskSize::Mask_5x5;
        }
        this->src1.FixedFilter(this->dst, FixedFilter::LowPass, ms, BorderType::Replicate, this->ctx);
    }

    FilterArea fa{3};

  private:
};

template <typename T> class LowPassFilterNpp : public TestNppSrcDstBase<T>
{
  public:
    LowPassFilterNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        NppiMaskSize ms = NPP_MASK_SIZE_3_X_3;
        if (fa.Size == 5)
        {
            ms = NPP_MASK_SIZE_5_X_5;
        }
        this->src1.FilterLowPassBorder(this->dst, ms, NPP_BORDER_REPLICATE, this->ctx);
    }

    FilterArea fa{3};

  private:
};

template <PixelType T, typename T2>
class LowPassFilterTest : public FilteringBase<LowPassFilterOpp<T>, LowPassFilterNpp<T2>>
{
  public:
    LowPassFilterTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : FilteringBase<LowPassFilterOpp<T>, LowPassFilterNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "LowPassFilter";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};

template <PixelType T> class GaussFilterOpp : public TestOppSrcDstBase<T>
{
  public:
    GaussFilterOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        MaskSize ms = MaskSize::Mask_3x3;
        if (fa.Size == 5)
        {
            ms = MaskSize::Mask_5x5;
        }
        if (fa.Size == 7)
        {
            ms = MaskSize::Mask_7x7;
        }
        if (fa.Size == 9)
        {
            ms = MaskSize::Mask_9x9;
        }
        if (fa.Size == 11)
        {
            ms = MaskSize::Mask_11x11;
        }
        if (fa.Size == 13)
        {
            ms = MaskSize::Mask_13x13;
        }
        if (fa.Size == 15)
        {
            ms = MaskSize::Mask_15x15;
        }
        this->src1.FixedFilter(this->dst, FixedFilter::Gauss, ms, BorderType::Replicate, this->ctx);
    }

    FilterArea fa{3};

  private:
};

template <typename T> class GaussFilterNpp : public TestNppSrcDstBase<T>
{
  public:
    GaussFilterNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        NppiMaskSize ms = NPP_MASK_SIZE_3_X_3;
        if (fa.Size == 5)
        {
            ms = NPP_MASK_SIZE_5_X_5;
        }
        if (fa.Size == 7)
        {
            ms = NPP_MASK_SIZE_7_X_7;
        }
        if (fa.Size == 9)
        {
            ms = NPP_MASK_SIZE_9_X_9;
        }
        if (fa.Size == 11)
        {
            ms = NPP_MASK_SIZE_11_X_11;
        }
        if (fa.Size == 13)
        {
            ms = NPP_MASK_SIZE_13_X_13;
        }
        if (fa.Size == 15)
        {
            ms = NPP_MASK_SIZE_15_X_15;
        }
        this->src1.FilterGaussBorder(this->dst, ms, NPP_BORDER_REPLICATE, this->ctx);
    }

    FilterArea fa{3};

  private:
};

template <PixelType T, typename T2> class GaussFilterTest : public FilteringBase<GaussFilterOpp<T>, GaussFilterNpp<T2>>
{
  public:
    GaussFilterTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : FilteringBase<GaussFilterOpp<T>, GaussFilterNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "GaussFilter";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};

template <PixelType T> class GaussAdvancedFilterOpp : public TestOppSrcDstBase<T>
{
  public:
    GaussAdvancedFilterOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.SeparableFilter(this->dst, filter, fa.Size.x, fa.Center.x, BorderType::Replicate, this->ctx);
    }

    FilterArea fa{3};
    DevVar<float> filter{3};

  private:
};

template <typename T> class GaussAdvancedFilterNpp : public TestNppSrcDstBase<T>
{
  public:
    GaussAdvancedFilterNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.FilterGaussAdvancedBorder(this->dst, fa.Size.x, filter, NPP_BORDER_REPLICATE, this->ctx);
    }

    FilterArea fa{3};
    DevVar<float> filter{3};

  private:
};

template <PixelType T, typename T2>
class GaussAdvancedFilterTest : public FilteringBase<GaussAdvancedFilterOpp<T>, GaussAdvancedFilterNpp<T2>>
{
  public:
    GaussAdvancedFilterTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : FilteringBase<GaussAdvancedFilterOpp<T>, GaussAdvancedFilterNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    template <typename ImgT> void Init(const ImgT &aCpuSrc1, const FilterArea &aFilterArea)
    {
        FilteringBase<GaussAdvancedFilterOpp<T>, GaussAdvancedFilterNpp<T2>>::Init(aCpuSrc1);

        std::vector<float> filter_h(to_size_t(aFilterArea.Size.x));
        this->opp.filter   = DevVar<float>(to_size_t(aFilterArea.Size.x));
        this->npp.filter   = DevVar<float>(to_size_t(aFilterArea.Size.x));
        const double sigma = 0.4 + (to_double(aFilterArea.Size.x) / 3.0) * 0.6;

        float sum_filter = 0;
        for (size_t i = 0; i < to_size_t(aFilterArea.Size.x); i++)
        {
            double x    = to_double(i) - to_double(aFilterArea.Size.x / 2);
            filter_h[i] = to_float(1.0 / (std::sqrt(2.0 * std::numbers::pi_v<double>) * sigma) *
                                   std::exp(-(x * x) / (2.0 * sigma * sigma)));
            sum_filter += filter_h[i];
        }
        for (size_t i = 0; i < to_size_t(aFilterArea.Size.x); i++)
        {
            filter_h[i] /= sum_filter;
        }

        this->opp.filter << filter_h;
        this->npp.filter << filter_h;
    }

    std::string GetName() override
    {
        return "GaussAdvancedFilter";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};
} // namespace opp