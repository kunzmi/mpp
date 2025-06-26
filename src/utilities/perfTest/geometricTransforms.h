#pragma once
#include "testNppBase.h"
#include "testOppBase.h"
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/filterArea.h>
#include <common/vector_typetraits.h>

namespace opp
{
template <typename oppT, typename nppT> class GeometricTransformBase
{
  public:
    GeometricTransformBase(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : opp(aIterations, aRepeats, aWidth, aHeight), npp(aIterations, aRepeats, aWidth, aHeight), repeats(aRepeats)
    {
    }
    virtual ~GeometricTransformBase() = default;

    GeometricTransformBase(const GeometricTransformBase &)     = default;
    GeometricTransformBase(GeometricTransformBase &&) noexcept = default;

    GeometricTransformBase &operator=(const GeometricTransformBase &)     = default;
    GeometricTransformBase &operator=(GeometricTransformBase &&) noexcept = default;

    virtual std::string GetName() = 0;
    virtual std::string GetType() = 0;

    template <typename ImgT> void Init(const ImgT &aCpuSrc1)
    {
        opp.Init();
        npp.Init();

        aCpuSrc1 >> opp.GetSrc1();
        aCpuSrc1 >> npp.GetSrc1();
        // set pixel outside of transform area to well defined values:
        aCpuSrc1 >> opp.GetDst();
        aCpuSrc1 >> npp.GetDst();
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
        std::stringstream ss;
        ss << GetName() << " for " << GetType() << " and interpolation mode " << opp.interpol;

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

template <PixelType T> class AffineTransformOpp : public TestOppSrcDstBase<T>
{
  public:
    AffineTransformOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
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
class AffineTransformTest : public GeometricTransformBase<AffineTransformOpp<T>, AffineTransformNpp<T2>>
{
  public:
    AffineTransformTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : GeometricTransformBase<AffineTransformOpp<T>, AffineTransformNpp<T2>>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    template <typename ImgT>
    void Init(const ImgT &aCpuSrc1, const AffineTransformation<double> &aTransform, InterpolationMode aInterpol)
    {
        GeometricTransformBase<AffineTransformOpp<T>, AffineTransformNpp<T2>>::Init(aCpuSrc1);

        this->opp.affine = aTransform;
        this->npp.affine = aTransform;

        this->opp.interpol = aInterpol;
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
};

template <PixelType T> class PerspectiveTransformOpp : public TestOppSrcDstBase<T>
{
  public:
    PerspectiveTransformOpp(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcDstBase<T>(aIterations, aRepeats, aWidth, aHeight)
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
class PerspectiveTransformTest : public GeometricTransformBase<PerspectiveTransformOpp<T>, PerspectiveTransformNpp<T2>>
{
  public:
    PerspectiveTransformTest(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : GeometricTransformBase<PerspectiveTransformOpp<T>, PerspectiveTransformNpp<T2>>(aIterations, aRepeats, aWidth,
                                                                                          aHeight)
    {
    }

    template <typename ImgT>
    void Init(const ImgT &aCpuSrc1, const PerspectiveTransformation<double> &aTransform, InterpolationMode aInterpol)
    {
        GeometricTransformBase<PerspectiveTransformOpp<T>, PerspectiveTransformNpp<T2>>::Init(aCpuSrc1);

        this->opp.perspective = aTransform;
        this->npp.perspective = aTransform;

        this->opp.interpol = aInterpol;
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
};
} // namespace opp