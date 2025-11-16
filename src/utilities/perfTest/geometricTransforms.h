#pragma once
#include "testMppBase.h"
#include "testNppBase.h"
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/filterArea.h>
#include <common/vector_typetraits.h>

namespace mpp
{
template <typename mppT, typename nppT> class GeometricTransformBase
{
  public:
    GeometricTransformBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : mpp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~GeometricTransformBase() = default;

    GeometricTransformBase(const GeometricTransformBase &)     = default;
    GeometricTransformBase(GeometricTransformBase &&) noexcept = default;

    GeometricTransformBase &operator=(const GeometricTransformBase &)     = default;
    GeometricTransformBase &operator=(GeometricTransformBase &&) noexcept = default;

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
        // set pixel outside of transform area to well defined values:
        aCpuSrc1 >> mpp.GetDst();
        aCpuSrc1 >> npp.GetDst();
    }

    void Run(const Roi &aRoi)
    {
        mpp.SetRoi(aRoi);
        npp.SetRoi(aRoi);

        // Who ever runs first, be it NPP or MPP, will be penalized with some outliers, likely due to screen updates...
        // So run twice and forget the first results.
        mpp.WarmUp();
        rt_mpp = mpp.Run();

        npp.WarmUp();
        rt_npp = npp.Run();

        mpp.WarmUp();
        rt_mpp = mpp.Run();
    }

    template <typename ImgT, PixelType resT> TestResult GetResult()
    {
        std::stringstream ss;
        ss << GetName() << " for " << GetType() << " and interpolation mode " << mpp.interpol;

        std::cout << ss.str() << std::endl;

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

        res.Name                   = ss.str();
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

template <PixelType T> class AffineTransformMpp : public TestMppSrcDstBase<T>
{
  public:
    AffineTransformMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.WarpAffine(this->dst, affine, interpol, BorderType::None, this->ctx);
    }

    InterpolationMode interpol{InterpolationMode::NearestNeighbor};
    AffineTransformation<double> affine;

  private:
};

template <typename T> class AffineTransformNpp : public TestNppSrcDstBase<T>
{
  public:
    AffineTransformNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.WarpAffine(this->dst, affine, interpol, this->ctx);
    }

    int interpol{static_cast<int>(NPPI_INTER_NN)};
    AffineTransformation<double> affine;

  private:
};

template <PixelType T, typename T2>
class AffineTransformTest : public GeometricTransformBase<AffineTransformMpp<T>, AffineTransformNpp<T2>>
{
  public:
    AffineTransformTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : GeometricTransformBase<AffineTransformMpp<T>, AffineTransformNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    template <typename ImgT>
    void Init(const ImgT &aCpuSrc1, const AffineTransformation<double> &aTransform, InterpolationMode aInterpol)
    {
        GeometricTransformBase<AffineTransformMpp<T>, AffineTransformNpp<T2>>::Init(aCpuSrc1);

        this->mpp.affine = aTransform;
        this->npp.affine = aTransform;

        this->mpp.interpol = aInterpol;
        this->npp.interpol = static_cast<int>(aInterpol);
    }

    std::string GetName() override
    {
        return "AffineTransformation";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
    int GetOrder1() override
    {
        return 400 + 0;
    }
    int GetOrder2() override
    {
        return this->npp.interpol;
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

template <PixelType T> class PerspectiveTransformMpp : public TestMppSrcDstBase<T>
{
  public:
    PerspectiveTransformMpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestMppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.WarpPerspective(this->dst, perspective, interpol, BorderType::None, this->ctx);
    }

    InterpolationMode interpol{InterpolationMode::NearestNeighbor};
    PerspectiveTransformation<double> perspective;

  private:
};

template <typename T> class PerspectiveTransformNpp : public TestNppSrcDstBase<T>
{
  public:
    PerspectiveTransformNpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestNppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void RunOnce() override
    {
        this->src1.WarpPerspective(this->dst, perspective, interpol, this->ctx);
    }

    int interpol{static_cast<int>(NPPI_INTER_NN)};
    PerspectiveTransformation<double> perspective;

  private:
};

template <PixelType T, typename T2>
class PerspectiveTransformTest : public GeometricTransformBase<PerspectiveTransformMpp<T>, PerspectiveTransformNpp<T2>>
{
  public:
    PerspectiveTransformTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : GeometricTransformBase<PerspectiveTransformMpp<T>, PerspectiveTransformNpp<T2>>(aIterations, aRepeats, aWidth,
                                                                                          aHeight)
    {
    }

    template <typename ImgT>
    void Init(const ImgT &aCpuSrc1, const PerspectiveTransformation<double> &aTransform, InterpolationMode aInterpol)
    {
        GeometricTransformBase<PerspectiveTransformMpp<T>, PerspectiveTransformNpp<T2>>::Init(aCpuSrc1);

        this->mpp.perspective = aTransform;
        this->npp.perspective = aTransform;

        this->mpp.interpol = aInterpol;
        this->npp.interpol = static_cast<int>(aInterpol);
    }

    std::string GetName() override
    {
        return "PerspectiveTransformation";
    }

    std::string GetType() override
    {
        return pixel_type_name<T>::value;
    }
    int GetOrder1() override
    {
        return 400 + 1;
    }
    int GetOrder2() override
    {
        return this->npp.interpol;
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