#pragma once
#include "testBase.h"
#include "testNppBase.h"
#include "testOppBase.h"
#include <backends/simple_cpu/image/imageView.h>
#include <common/vector_typetraits.h>

namespace opp
{
template <typename oppT, typename nppT> class ArithmeticBase
{
  public:
    ArithmeticBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : opp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~ArithmeticBase() = default;

    ArithmeticBase(const ArithmeticBase &)     = default;
    ArithmeticBase(ArithmeticBase &&) noexcept = default;

    ArithmeticBase &operator=(const ArithmeticBase &)     = default;
    ArithmeticBase &operator=(ArithmeticBase &&) noexcept = default;

    virtual std::string GetName() = 0;
    virtual std::string GetType() = 0;

    template <typename ImgT> void Init(const ImgT &aCpuSrc1, const ImgT &aCpuSrc2)
    {
        opp.Init();
        npp.Init();

        aCpuSrc1 >> opp.GetSrc1();
        aCpuSrc2 >> opp.GetSrc2();

        aCpuSrc1 >> npp.GetSrc1();
        aCpuSrc2 >> npp.GetSrc2();
    }

    void Run(const Roi &aRoi)
    {
        opp.SetRoi(aRoi);
        npp.SetRoi(aRoi);

        opp.WarmUp();
        rt_opp = opp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();
    }

    template <typename ImgT, PixelType resT> TestResult GetResult()
    {
        std::cout << GetName() << " for " << GetType() << std::endl;

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

  private:
    oppT opp;
    nppT npp;

    Runtime rt_opp;
    Runtime rt_npp;

    size_t repeats;
};

template <PixelType T> class AddOpp : public TestOppSrcSrcDstBase<T>
{
  public:
    AddOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<T>)
        {
            this->src1.Add(this->src2, this->dst, 0, this->ctx);
        }
        else
        {
            this->src1.Add(this->src2, this->dst, this->ctx);
        }
    }
};

template <typename T> class AddNpp : public TestNppSrcSrcDstBase<T>
{
  public:
    AddNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<typename T::pixel_type_t>)
        {
            this->src1.Add(this->src2, this->dst, 0, this->ctx);
        }
        else
        {
            this->src1.Add(this->src2, this->dst, this->ctx);
        }
    }
};

template <PixelType T, typename T2> class AddTest : public ArithmeticBase<AddOpp<T>, AddNpp<T2>>
{
  public:
    AddTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<AddOpp<T>, AddNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "Add";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};

template <PixelType T> class SubOpp : public TestOppSrcSrcDstBase<T>
{
  public:
    SubOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<T>)
        {
            this->src1.Sub(this->src2, this->dst, 0, this->ctx);
        }
        else
        {
            this->src1.Sub(this->src2, this->dst, this->ctx);
        }
    }
};

template <typename T> class SubNpp : public TestNppSrcSrcDstBase<T>
{
  public:
    SubNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<typename T::pixel_type_t>)
        {
            this->src2.Sub(this->src1, this->dst, 0, this->ctx);
        }
        else
        {
            this->src2.Sub(this->src1, this->dst, this->ctx);
        }
    }
};

template <PixelType T, typename T2> class SubTest : public ArithmeticBase<SubOpp<T>, SubNpp<T2>>
{
  public:
    SubTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<SubOpp<T>, SubNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "Sub";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};

template <PixelType T> class MulOpp : public TestOppSrcSrcDstBase<T>
{
  public:
    MulOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<T>)
        {
            this->src1.Mul(this->src2, this->dst, -1, this->ctx);
        }
        else
        {
            this->src1.Mul(this->src2, this->dst, this->ctx);
        }
    }
};

template <typename T> class MulNpp : public TestNppSrcSrcDstBase<T>
{
  public:
    MulNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<typename T::pixel_type_t>)
        {
            this->src1.Mul(this->src2, this->dst, -1, this->ctx);
        }
        else
        {
            this->src1.Mul(this->src2, this->dst, this->ctx);
        }
    }
};

template <PixelType T, typename T2> class MulTest : public ArithmeticBase<MulOpp<T>, MulNpp<T2>>
{
  public:
    MulTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<MulOpp<T>, MulNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "Mul";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};

template <PixelType T> class DivOpp : public TestOppSrcSrcDstBase<T>
{
  public:
    DivOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<T>)
        {
            this->src1.Div(this->src2, this->dst, 0, RoundingMode::NearestTiesToEven, this->ctx);
        }
        else
        {
            this->src1.Div(this->src2, this->dst, this->ctx);
        }
    }
};

template <typename T> class DivNpp : public TestNppSrcSrcDstBase<T>
{
  public:
    DivNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        if constexpr (RealOrComplexIntVector<typename T::pixel_type_t>)
        {
            this->src2.Div(this->src1, this->dst, 0, this->ctx);
        }
        else
        {
            this->src2.Div(this->src1, this->dst, this->ctx);
        }
    }
};

template <PixelType T, typename T2> class DivTest : public ArithmeticBase<DivOpp<T>, DivNpp<T2>>
{
  public:
    DivTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<DivOpp<T>, DivNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    std::string GetName() override
    {
        return "Div";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
};
} // namespace opp