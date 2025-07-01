#pragma once
#include "testBase.h"
#include "testMppBase.h"
#include "testNppBase.h"
#include <backends/simple_cpu/image/imageView.h>
#include <common/vector_typetraits.h>

namespace mpp
{
template <typename mppT, typename nppT> class ArithmeticBase
{
  public:
    ArithmeticBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : mpp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~ArithmeticBase() = default;

    ArithmeticBase(const ArithmeticBase &)     = default;
    ArithmeticBase(ArithmeticBase &&) noexcept = default;

    ArithmeticBase &operator=(const ArithmeticBase &)     = default;
    ArithmeticBase &operator=(ArithmeticBase &&) noexcept = default;

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
        aCpuSrc2 >> mpp.GetSrc2();

        aCpuSrc1 >> npp.GetSrc1();
        aCpuSrc2 >> npp.GetSrc2();
    }

    void Run(const Roi &aRoi)
    {
        mpp.SetRoi(aRoi);
        npp.SetRoi(aRoi);

        mpp.WarmUp();
        rt_mpp = mpp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();
    }

    template <typename ImgT, PixelType resT> TestResult GetResult()
    {
        std::cout << GetName() << " for " << GetType() << std::endl;

        ImgT resNpp(npp.GetDst().SizeAlloc());
        ImgT resMpp(mpp.GetDst().SizeAlloc());

        resNpp << npp.GetDst();
        resMpp << mpp.GetDst();

        resNpp.SetRoi(npp.GetDst().ROI());
        resMpp.SetRoi(mpp.GetDst().ROI());

        double maxErrorScalar = 0;
        resT maxError{0};

        if constexpr (vector_size_v<resT> == 1)
        {
            resMpp.MaximumError(resNpp, maxError);
            maxErrorScalar = maxError.x;
        }
        else
        {
            resMpp.MaximumError(resNpp, maxError, maxErrorScalar);
        }
        double avgErrorScalar = 0;
        resT avgError{0};

        if constexpr (vector_size_v<resT> == 1)
        {
            resMpp.AverageError(resNpp, avgError);
            avgErrorScalar = avgError.x;
        }
        else
        {
            resMpp.AverageError(resNpp, avgError, avgErrorScalar);
        }

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

        std::cout << "MaxError: " << maxErrorScalar << " AvgError: " << avgErrorScalar << std::endl;
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

  private:
    mppT mpp;
    nppT npp;

    Runtime rt_mpp;
    Runtime rt_npp;

    size_t repeats;
};

template <PixelType T> class AddMpp : public TestMppSrcSrcDstBase<T>
{
  public:
    AddMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
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

template <PixelType T, typename T2> class AddTest : public ArithmeticBase<AddMpp<T>, AddNpp<T2>>
{
  public:
    AddTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<AddMpp<T>, AddNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
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
    int GetOrder1() override
    {
        return 100 + 0;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<T>)
        {
            return channel_count_v<T> + 64 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else if constexpr (RealSignedVector<T>)
        {
            return channel_count_v<T> + 32 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else
        {
            return channel_count_v<T> + sizeof(pixel_basetype_t<T>) * 128;
        }
    }
};

template <PixelType T> class SubMpp : public TestMppSrcSrcDstBase<T>
{
  public:
    SubMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
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

template <PixelType T, typename T2> class SubTest : public ArithmeticBase<SubMpp<T>, SubNpp<T2>>
{
  public:
    SubTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<SubMpp<T>, SubNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
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
    int GetOrder1() override
    {
        return 100 + 1;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<T>)
        {
            return channel_count_v<T> + 64 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else if constexpr (RealSignedVector<T>)
        {
            return channel_count_v<T> + 32 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else
        {
            return channel_count_v<T> + sizeof(pixel_basetype_t<T>) * 128;
        }
    }
};

template <PixelType T> class MulMpp : public TestMppSrcSrcDstBase<T>
{
  public:
    MulMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
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

template <PixelType T, typename T2> class MulTest : public ArithmeticBase<MulMpp<T>, MulNpp<T2>>
{
  public:
    MulTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<MulMpp<T>, MulNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
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
    int GetOrder1() override
    {
        return 100 + 2;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<T>)
        {
            return channel_count_v<T> + 64 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else if constexpr (RealSignedVector<T>)
        {
            return channel_count_v<T> + 32 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else
        {
            return channel_count_v<T> + sizeof(pixel_basetype_t<T>) * 128;
        }
    }
};

template <PixelType T> class DivMpp : public TestMppSrcSrcDstBase<T>
{
  public:
    DivMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
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

template <PixelType T, typename T2> class DivTest : public ArithmeticBase<DivMpp<T>, DivNpp<T2>>
{
  public:
    DivTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : ArithmeticBase<DivMpp<T>, DivNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
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
    int GetOrder1() override
    {
        return 100 + 3;
    }
    int GetOrder2() override
    {
        return 0;
    }
    int GetOrder3() override
    {
        if constexpr (RealOrComplexFloatingVector<T>)
        {
            return channel_count_v<T> + 64 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else if constexpr (RealSignedVector<T>)
        {
            return channel_count_v<T> + 32 + sizeof(pixel_basetype_t<T>) * 128;
        }
        else
        {
            return channel_count_v<T> + sizeof(pixel_basetype_t<T>) * 128;
        }
    }
};
} // namespace mpp