#include "backends/npp/image/imageView.h"
#include "common/exception.h"
#include "common/image/pixelTypes.h"
#include <algorithm>
#include <backends/cuda/devVar.h>     // NOLINT(misc-include-cleaner)
#include <backends/cuda/devVarView.h> // NOLINT(misc-include-cleaner)
#include <backends/cuda/event.h>
#include <backends/cuda/image/image.h>     // NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView.h> // NOLINT(misc-include-cleaner)
#include <backends/cuda/image/statistics/integralSqr.h>
#include <backends/cuda/image/statistics/ssim.h>
#include <backends/cuda/stream.h>
#include <backends/cuda/streamCtx.h>
#include <backends/npp/image/image16s.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image16sC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image16u.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image16uC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32f.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image32s.h> // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image8u.h>       // NOLINT(misc-include-cleaner)
#include <backends/npp/image/image8uC1View.h> // NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <cmath>
#include <common/image/functors/reductionInitValues.h>
#include <common/scratchBuffer.h>
#include <common/version.h>
#include <cstddef>
#include <filesystem> // NOLINT(misc-include-cleaner)
#include <ios>
#include <iostream>
#include <nppcore.h>
#include <nppdefs.h>
#include <numeric>
#include <vector>

#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <fstream>
#include <iostream>
#include <numbers>

#include "pixel8uC1.h"
#include "pixel8uC3.h"
#include "pixel8uC4.h"

#include "pixel16uC1.h"
#include "pixel16uC3.h"
#include "pixel16uC4.h"

#include "pixel32fC1.h"
#include "pixel32fC3.h"
#include "pixel32fC4.h"

#include "pixel32sC1.h"

using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace opp::image::cuda;
namespace nv  = opp::image::npp;
namespace cpu = opp::image::cpuSimple;
/*
struct Runtime
{
    float Min{0};
    float Max{0};
    float Mean{0};
    float Std{0};
    float Total{0};

    Runtime(const std::vector<float> &aTimings, float aTotal)
    {
        Total = aTotal;
        Min   = *std::min_element(aTimings.begin(), aTimings.end());
        Max   = *std::max_element(aTimings.begin(), aTimings.end());

        const float sum    = std::accumulate(aTimings.begin(), aTimings.end(), 0.0f);
        const float sumSqr = std::inner_product(aTimings.begin(), aTimings.end(), aTimings.begin(), 0.0f);

        size_t iterations = aTimings.size();
        if (iterations > 1)
        {
            Std  = std::sqrt((sumSqr - (sum * sum) / iterations) / (iterations - 1.0f));
            Mean = sum / iterations;
        }
    }
};

std::ostream &operator<<(std::ostream &aOs, const Runtime &aRuntime)
{
    aOs << "Total: " << aRuntime.Total << " ms - Mean: " << aRuntime.Mean << " Std: " << aRuntime.Std
        << " Min: " << aRuntime.Min << " Max: " << aRuntime.Max << std::endl;

    return aOs;
}

class TestBase
{
  public:
    TestBase(size_t aIterations, size_t aRepeats) : iterations(aIterations), repeats(aRepeats), runtimes(aIterations, 0)
    {
    }
    virtual ~TestBase() = default;

    TestBase(const TestBase &)     = default;
    TestBase(TestBase &&) noexcept = default;

    TestBase &operator=(const TestBase &)     = default;
    TestBase &operator=(TestBase &&) noexcept = default;

    Runtime Run()
    {
        startGlobal.Record();
        for (size_t i = 0; i < iterations; i++)
        {
            startIter.Record();
            for (size_t r = 0; r < repeats; r++)
            {
                RunOnce();
            }
            endIter.Record();
            endIter.Synchronize();
            runtimes[i] = endIter - startIter;
        }
        endGlobal.Record();
        endGlobal.Synchronize();
        totalRuntime = endGlobal - startGlobal;

        return Runtime(runtimes, totalRuntime);
    }

    void WarmUp()
    {
        RunOnce();
    }

    Runtime GetRuntime()
    {
        return Runtime(runtimes, totalRuntime);
    }

  protected:
    virtual void RunOnce() = 0;

  private:
    size_t iterations;
    size_t repeats;
    std::vector<float> runtimes;
    float totalRuntime{0};
    Event startGlobal;
    Event startIter;
    Event endIter;
    Event endGlobal;
};

template <typename T> class TestOPPAdd : public TestBase
{
  public:
    TestOPPAdd(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight), dst(aWidth, aHeight)
    {
        ctx = StreamCtxSingleton::Get();
    }

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

  private:
    Image<T> src1;
    Image<T> src2;
    Image<T> dst;

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.Add(src2, dst, 0, ctx);
    }
};

template <typename T> class TestNPPAdd : public TestBase
{
  public:
    TestNPPAdd(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight), dst(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
    }

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

  private:
    T src1;
    T src2;
    T dst;

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.Add(src2, dst, 0, ctx);
    }
};

template <typename T> class TestOPPQualityIndex : public TestBase
{
  public:
    TestOPPQualityIndex(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight)
    {
        ctx    = StreamCtxSingleton::Get();
        buffer = DevVar<byte>(src1.QualityIndexBufferSize(ctx));
    }

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

    void SetSrc(const cpu::Image<T> &aImg1, const cpu::Image<T> &aImg2)
    {
        aImg1 >> src1;
        aImg2 >> src2;
    }

  private:
    Image<T> src1;
    Image<T> src2;
    DevVar<byte> buffer{0};
    DevVar<Pixel64fC1> dst{1};

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.QualityIndex(src2, dst, buffer, ctx);
    }
};

template <typename T> class TestNPPQualityIndex : public TestBase
{
  public:
    TestNPPQualityIndex(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), src2(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
        buffer = DevVar<byte>(src1.QualityIndexGetBufferHostSize(ctx));
    }

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

    template <typename T2> void SetSrc(const T2 &aImg1, const T2 &aImg2)
    {
        aImg1 >> src1;
        aImg2 >> src2;
    }

  private:
    T src1;
    T src2;
    DevVar<byte> buffer{0};
    DevVar<float> dst{1};

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.QualityIndex(src2, src1.NppiSizeRoi(), dst, buffer, ctx);
    }
};

template <typename T> class TestOPPMean : public TestBase
{
  public:
    TestOPPMean(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
        ctx    = StreamCtxSingleton::Get();
        buffer = DevVar<byte>(src1.MeanBufferSize(ctx));
    }

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
    }

  private:
    Image<T> src1;
    DevVar<byte> buffer{0};
    DevVar<Pixel64fC1> mean{1};

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.Mean(mean, buffer, ctx);
    }
};

template <typename T> class TestNPPMean : public TestBase
{
  public:
    TestNPPMean(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
        buffer = DevVar<byte>(src1.MeanGetBufferHostSize(ctx));
    }

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
    }

  private:
    T src1;
    DevVar<byte> buffer{0};
    DevVar<double> mean{1};

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.Mean(buffer, mean, ctx);
    }
};

template <typename T> class TestOPPMinMaxIdx : public TestBase
{
  public:
    TestOPPMinMaxIdx(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
        ctx    = StreamCtxSingleton::Get();
        buffer = DevVar<byte>(src1.MinMaxIndexBufferSize(ctx));
    }

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
    }

  private:
    Image<T> src1;
    DevVar<byte> buffer{0};
    DevVar<T> min{1};
    DevVar<T> max{1};
    DevVar<remove_vector_t<T>> minScalar{1};
    DevVar<remove_vector_t<T>> maxScalar{1};
    DevVar<IndexMinMax> idx1{1};
    DevVar<IndexMinMaxChannel> idx2{1};

    StreamCtx ctx;

    void RunOnce() override
    {
        if constexpr (vector_size_v<T> == 1)
        {
            src1.MinMaxIndex(min, max, idx1, buffer, ctx);
        }
        else
        {
            src1.MinMaxIndex(min, max, idx1, minScalar, maxScalar, idx2, buffer, ctx);
        }
    }
};

template <typename T> class TestNPPMinMaxIdx : public TestBase
{
  public:
    TestNPPMinMaxIdx(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
        buffer = DevVar<byte>(src1.MeanGetBufferHostSize(ctx));
    }

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
    }

  private:
    T src1;
    DevVar<byte> buffer{0};
    DevVar<byte> min{4};
    DevVar<byte> max{4};
    DevVar<NppiPoint> idxMin{4};
    DevVar<NppiPoint> idxMax{4};

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.MinMaxIndx(min, max, idxMax, idxMin, buffer, ctx);
    }
};

template <typename T> class TestOPPMeanStd : public TestBase
{
  public:
    TestOPPMeanStd(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
        ctx    = StreamCtxSingleton::Get();
        buffer = DevVar<byte>(src1.MeanStdBufferSize(ctx));
    }

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
    }

  private:
    Image<T> src1;
    DevVar<byte> buffer{0};
    DevVar<Pixel64fC1> mean{1};
    DevVar<Pixel64fC1> std{1};

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.MeanStd(mean, std, buffer, ctx);
    }
};

template <typename T> class TestNPPMeanStd : public TestBase
{
  public:
    TestNPPMeanStd(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
        buffer = DevVar<byte>(src1.MeanStdDevGetBufferHostSize(ctx));
    }

    void SetRoi(const Roi &aRoi)
    {
        src1.SetRoi(aRoi);
    }

    void SetBorder(const Border &aBorder)
    {
        src1.SetRoi(aBorder);
    }

  private:
    T src1;
    DevVar<byte> buffer{0};
    DevVar<double> mean{1};
    DevVar<double> std{1};

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.Mean_StdDev(buffer, mean, std, ctx);
    }
};

template <typename T> class TestOPPFilterGauss : public TestBase
{
  public:
    TestOPPFilterGauss(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        ctx = StreamCtxSingleton::Get();
    }

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

  private:
    Image<T> src1;
    Image<T> dst;

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.FixedFilter(dst, FixedFilter::Gauss, MaskSize::Mask_15x15, BorderType::Replicate, ctx);
    }
};

template <typename T> class TestNPPFilterGauss : public TestBase
{
  public:
    TestNPPFilterGauss(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
    }

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

  private:
    T src1;
    T dst;

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.FilterGaussBorder(dst, NPP_MASK_SIZE_15_X_15, NPP_BORDER_REPLICATE, ctx);
    }
};

template <typename T> class TestOPPFilterGaussAdvanced : public TestBase
{
  public:
    TestOPPFilterGaussAdvanced(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        ctx = StreamCtxSingleton::Get();
        std::vector<float> temp(filter.Size(), 1.0f / static_cast<float>(filter.Size()));
        filter << temp;
    }

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

  private:
    Image<T> src1;
    Image<T> dst;
    DevVar<float> filter{9};

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.SeparableFilter(dst, filter, 9, 4, BorderType::Replicate, ctx);
    }
};

template <typename T> class TestNPPFilterGaussAdvanced : public TestBase
{
  public:
    TestNPPFilterGaussAdvanced(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
        std::vector<float> temp(filter.Size(), 1.0f / static_cast<float>(filter.Size()));
        filter << temp;
    }

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

  private:
    T src1;
    T dst;
    DevVar<float> filter{9};

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.FilterGaussAdvancedBorder(dst, 9, filter, NPP_BORDER_REPLICATE, ctx);
    }
};

template <typename T> class TestOPPFilter : public TestBase
{
  public:
    TestOPPFilter(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        ctx = StreamCtxSingleton::Get();
        std::vector<float> temp(filter.Size(), 1.0f / static_cast<float>(filter.Size()));
        filter << temp;
    }

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

  private:
    Image<T> src1;
    Image<T> dst;
    FilterArea filterArea{9};
    DevVar<float> filter{9 * 9};

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.Filter(dst, filter, filterArea, BorderType::Replicate, ctx);
    }
};

template <typename T> class TestNPPFilter : public TestBase
{
  public:
    TestNPPFilter(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
        std::vector<float> temp(filter.Size(), 1.0f / static_cast<float>(filter.Size()));
        filter << temp;
    }

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

  private:
    T src1;
    T dst;
    DevVar<float> filter{9 * 9};

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.FilterBorder32f(dst, filter, {9, 9}, {4, 4}, NPP_BORDER_REPLICATE, ctx);
    }
};

template <typename T> class TestOPPFilterLaplace : public TestBase
{
  public:
    TestOPPFilterLaplace(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        ctx = StreamCtxSingleton::Get();
    }

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

  private:
    Image<T> src1;
    Image<T> dst;

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.FixedFilter(dst, FixedFilter::Laplace, MaskSize::Mask_5x5, BorderType::Replicate, ctx);
    }
};

template <typename T> class TestNPPFilterLaplace : public TestBase
{
  public:
    TestNPPFilterLaplace(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
    }

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

  private:
    T src1;
    T dst;

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.FilterLaplaceBorder(dst, NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, ctx);
    }
};

template <typename T> class TestOPPFilterBox : public TestBase
{
  public:
    TestOPPFilterBox(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        ctx = StreamCtxSingleton::Get();
    }

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

  private:
    Image<T> src1;
    Image<T> dst;

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.BoxFilter(dst, 11, BorderType::Replicate, ctx);
    }
};

template <typename T> class TestNPPFilterBox : public TestBase
{
  public:
    TestNPPFilterBox(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
    }

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

  private:
    T src1;
    T dst;

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.FilterBoxBorder(dst, {11, 11}, {5, 5}, NPP_BORDER_REPLICATE, ctx);
    }
};

template <typename T> class TestOPPAffineTransform : public TestBase
{
  public:
    TestOPPAffineTransform(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        ctx                                 = StreamCtxSingleton::Get();
        AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation(Vec2d(-aWidth / 2));
        AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
        AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation(Vec2d(aWidth / 2));
        affine                              = shift2 * rot * shift1;
    }

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

  private:
    Image<T> src1;
    Image<T> dst;
    AffineTransformation<double> affine;

    StreamCtx ctx;

    void RunOnce() override
    {
        src1.WarpAffine(dst, affine, InterpolationMode::CubicLagrange, BorderType::None, ctx);
    }
};

template <typename T> class TestNPPAffineTransform : public TestBase
{
  public:
    TestNPPAffineTransform(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestBase(aIterations, aRepeats), src1(aWidth, aHeight), dst(aWidth, aHeight)
    {
        NppStatus status = nppGetStreamContext(&ctx);
        if (status != NPP_SUCCESS)
        {
            throw NPPEXCEPTION("Failed to get context.");
        }
        AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation(Vec2d(-aWidth / 2));
        AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
        AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation(Vec2d(aWidth / 2));
        affine                              = shift2 * rot * shift1;
    }

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

  private:
    T src1;
    T dst;
    AffineTransformation<double> affine;

    NppStreamContext ctx;

    void RunOnce() override
    {
        src1.WarpAffine(dst, affine, static_cast<int>(NPPI_INTER_CUBIC), ctx);
    }
};

void print(cpu::Image<Pixel32fC1> &img)
{
    const int startX = 6;
    const int endX   = 8;
    const int startY = startX;
    const int endY   = endX;

    for (int y = startY; y <= endY; y++)
    {
        for (int x = startX; x <= endX; x++)
        {
            std::cout << img(x, y).x << " ";
        }
        std::cout << std::endl;
    }
}

template <PixelType T> class OppQualityIndex : public TestOppSrcSrcReductionBase<T, Pixel64fC1>
{
  public:
    OppQualityIndex(size_t aIterations, size_t aRepeats, int aWidth, int aHeight)
        : TestOppSrcSrcReductionBase<T, Pixel64fC1>(aIterations, aRepeats, aWidth, aHeight)
    {
    }

    void Init() override
    {
    }

  private:
    void RunOnce() override
    {
        this->src1.QualityIndex(this->src2, this->res, this->buffer, this->ctx);
    }
};*/

Border GetBorder()
{
    return Border(-1);
}

int main()
{
    try
    {
        const Border &ab = GetBorder();
        const Border &ac = {-1};
        if (ab == ac)
        {
            std::cout << "yep";
        }

        std::cout << "Hello world! This is " << OPP_PROJECT_NAME << " version " << OPP_VERSION << "!" << std::endl;

        constexpr size_t iterations = 100;
        constexpr size_t repeats    = 10; /**/
        constexpr int imgWidth      = 4096;
        constexpr int imgHeight     = 4096;
        Border b                    = Border(-1, 0, -1, 0);

        std::ofstream csv("results.csv", std::ofstream::out);

        runPixel8uC1(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel8uC3(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel8uC4(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;

        runPixel16uC1(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel16uC3(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel16uC4(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;

        runPixel32fC1(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel32fC3(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel32fC4(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        // runPixel8uC1(iterations, repeats, imgWidth, imgHeight, -1);

        runPixel32sC1(iterations, repeats, imgWidth, imgHeight, b, csv);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;

        Image<Pixel16uC1> ttt(imgWidth, imgHeight);
        nv::Image16uC3 ttt2(imgWidth, imgHeight);
        // ttt2.Max()
        //     ttt.M()

        /* Roi roi(0, 0, imgWidth, imgHeight);
        FilterArea filterArea(11);
        FilterArea filterAreaCol({1, 11}, {0, 5});
        FilterArea filterAreaRow({11, 1}, {5, 0});
        AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation(Vec2d(-imgWidth / 2));
        AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(30);
        AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation(Vec2d(imgWidth / 2));
        AffineTransformation<double> affine = shift2 * rot * shift1;
        PerspectiveTransformation<double> perspective(affine);
        perspective(2, 0) = 0.0002;
        perspective(2, 1) = -0.0003;

        AddTest<Pixel8uC1, nv::Image8uC1> testAdd(iterations, repeats, imgWidth, imgHeight);
        SubTest<Pixel8uC1, nv::Image8uC1> testSub(iterations, repeats, imgWidth, imgHeight);
        MulTest<Pixel8uC1, nv::Image8uC1> testMul(iterations, repeats, imgWidth, imgHeight);
        DivTest<Pixel8uC1, nv::Image8uC1> testDiv(iterations, repeats, imgWidth, imgHeight);

        BoxFilterTest<Pixel8uC1, nv::Image8uC1> testBoxFilter(iterations, repeats, imgWidth, imgHeight);
        RowWindowSumTest<Pixel8uC1, Pixel32fC1, nv::Image8uC1, nv::Image32fC1> testRowWindowSum(iterations, repeats,
                                                                                                imgWidth, imgHeight);
        ColumnWindowSumTest<Pixel8uC1, Pixel32fC1, nv::Image8uC1, nv::Image32fC1> testColumnWindowSum(
            iterations, repeats, imgWidth, imgHeight);
        LowPassFilterTest<Pixel8uC1, nv::Image8uC1> testLowPassFilter(iterations, repeats, imgWidth, imgHeight);
        GaussFilterTest<Pixel8uC1, nv::Image8uC1> testGaussFilter(iterations, repeats, imgWidth, imgHeight);
        GaussAdvancedFilterTest<Pixel8uC1, nv::Image8uC1> testGaussAdvancedilter(iterations, repeats, imgWidth,
                                                                                 imgHeight);

        AffineTransformTest<Pixel8uC1, nv::Image8uC1> testAffineTransformation(iterations, repeats, imgWidth,
                                                                               imgHeight);

        PerspectiveTransformTest<Pixel8uC1, nv::Image8uC1> testPerspectiveTransformation(iterations, repeats, imgWidth,
                                                                                         imgHeight);

        QualityIndexTest<Pixel8uC1, Pixel64fC1, nv::Image8uC1, float, vector_active_size_v<Pixel8uC1>> testQualityIndex(
            iterations, repeats, imgWidth, imgHeight);

        MaxTest<Pixel8uC1, Pixel8uC1, nv::Image8uC1, byte, vector_active_size_v<Pixel8uC1>> testMax(
            iterations, repeats, imgWidth, imgHeight);

        MeanTest<Pixel8uC1, Pixel64fC1, nv::Image8uC1, double, vector_active_size_v<Pixel8uC1>> testMean(
            iterations, repeats, imgWidth, imgHeight);

        MeanStdTest<Pixel8uC1, Pixel64fC1, nv::Image8uC1, double, vector_active_size_v<Pixel8uC1>> testMeanStd(
            iterations, repeats, imgWidth, imgHeight);

        MSETest<Pixel8uC1, Pixel64fC1, nv::Image8uC1, float, vector_active_size_v<Pixel8uC1>> testMSE(
            iterations, repeats, imgWidth, imgHeight);

        Image<Pixel8uC3> ttt(imgWidth, imgHeight);

        // ttt.MSE()
        nv::Image8uC1 ttt2(imgWidth, imgHeight);
        // ttt2.MSE(, )
        //     ttt2.SumWindowColumnBorder(,)
        cpu::Image<Pixel8uC1> cpu_src1(imgWidth, imgHeight);
        cpu::Image<Pixel8uC1> cpu_src2(imgWidth, imgHeight);
        cpu_src1.FillRandom(0);
        cpu_src2.FillRandom(1);
        cpu_src2.Add({1});

        testAdd.Init(cpu_src1, cpu_src2);
        testAdd.Run(roi);
        testAdd.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testSub.Init(cpu_src1, cpu_src2);
        testSub.Run(roi);
        testSub.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testMul.Init(cpu_src1, cpu_src2);
        testMul.Run(roi);
        testMul.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testDiv.Init(cpu_src1, cpu_src2);
        testDiv.Run(roi);
        testDiv.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testBoxFilter.Init(cpu_src1);
        testBoxFilter.Run(roi, filterArea);
        testBoxFilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testRowWindowSum.Init(cpu_src1);
        testRowWindowSum.Run(roi, filterAreaRow);
        testRowWindowSum.GetResult<cpu::Image<Pixel32fC1>, Pixel64fC1>();

        testColumnWindowSum.Init(cpu_src1);
        testColumnWindowSum.Run(roi, filterAreaCol);
        testColumnWindowSum.GetResult<cpu::Image<Pixel32fC1>, Pixel64fC1>();

        testLowPassFilter.Init(cpu_src1);
        testLowPassFilter.Run(roi, 3);
        testLowPassFilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testLowPassFilter.Init(cpu_src1);
        testLowPassFilter.Run(roi, 5);
        testLowPassFilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 3);
        testGaussFilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 5);
        testGaussFilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testGaussFilter.Init(cpu_src1);
        testGaussFilter.Run(roi, 15);
        testGaussFilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testGaussAdvancedilter.Init(cpu_src1, 9);
        testGaussAdvancedilter.Run(roi, 9);
        testGaussAdvancedilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testGaussAdvancedilter.Init(cpu_src1, 11);
        testGaussAdvancedilter.Run(roi, 11);
        testGaussAdvancedilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testGaussAdvancedilter.Init(cpu_src1, 21);
        testGaussAdvancedilter.Run(roi, 21);
        testGaussAdvancedilter.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::NearestNeighbor);
        testAffineTransformation.Run(roi);
        testAffineTransformation.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::Linear);
        testAffineTransformation.Run(roi);
        testAffineTransformation.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testAffineTransformation.Init(cpu_src1, affine, InterpolationMode::CubicLagrange);
        testAffineTransformation.Run(roi);
        testAffineTransformation.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::NearestNeighbor);
        testPerspectiveTransformation.Run(roi);
        testPerspectiveTransformation.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::Linear);
        testPerspectiveTransformation.Run(roi);
        testPerspectiveTransformation.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testPerspectiveTransformation.Init(cpu_src1, perspective, InterpolationMode::CubicLagrange);
        testPerspectiveTransformation.Run(roi);
        testPerspectiveTransformation.GetResult<cpu::Image<Pixel8uC1>, Pixel64fC1>();

        testQualityIndex.Init(cpu_src1, cpu_src2);
        testQualityIndex.Run(roi);
        testQualityIndex.GetResult<Pixel64fC1, Pixel32fC1>();

        testMax.Init(cpu_src1);
        testMax.Run(roi);
        testMax.GetResult<Pixel8uC1, Pixel8uC1>();

        testMean.Init(cpu_src1);
        testMean.Run(roi);
        testMean.GetResult<Pixel64fC1, Pixel64fC1>();

        testMeanStd.Init(cpu_src1);
        testMeanStd.Run(roi);
        testMeanStd.GetResult<Pixel64fC1, Pixel64fC1>();

        testMSE.Init(cpu_src1, cpu_src2);
        testMSE.Run(roi);
        testMSE.GetResult<Pixel64fC1, Pixel32fC1>();

        AddTest<Pixel32fC1, nv::Image32fC1> testAdd2(iterations, repeats, imgWidth, imgHeight);
        SubTest<Pixel32fC1, nv::Image32fC1> testSub2(iterations, repeats, imgWidth, imgHeight);
        MulTest<Pixel32fC1, nv::Image32fC1> testMul2(iterations, repeats, imgWidth, imgHeight);
        DivTest<Pixel32fC1, nv::Image32fC1> testDiv2(iterations, repeats, imgWidth, imgHeight);

        BoxFilterTest<Pixel32fC1, nv::Image32fC1> testBoxFilter2(iterations, repeats, imgWidth, imgHeight);

        cpu::Image<Pixel32fC1> cpu_src12(imgWidth, imgHeight);
        cpu::Image<Pixel32fC1> cpu_src22(imgWidth, imgHeight);
        cpu_src12.FillRandom(0);
        cpu_src22.FillRandom(1);
        cpu_src22.Add({1});

        testAdd2.Init(cpu_src12, cpu_src22);
        testAdd2.Run(roi);
        testAdd2.GetResult<cpu::Image<Pixel32fC1>, Pixel64fC1>();

        testSub2.Init(cpu_src12, cpu_src22);
        testSub2.Run(roi);
        testSub2.GetResult<cpu::Image<Pixel32fC1>, Pixel64fC1>();

        testMul2.Init(cpu_src12, cpu_src22);
        testMul2.Run(roi);
        testMul2.GetResult<cpu::Image<Pixel32fC1>, Pixel64fC1>();

        testDiv2.Init(cpu_src12, cpu_src22);
        testDiv2.Run(roi);
        testDiv2.GetResult<cpu::Image<Pixel32fC1>, Pixel64fC1>();

        testBoxFilter2.Init(cpu_src12);
        testBoxFilter2.Run(roi, filterArea);
        testBoxFilter2.GetResult<cpu::Image<Pixel32fC1>, Pixel64fC1>();*/

        // constexpr int sizeTpl       = 16;
        //  INT_MAX / 1024 / 256 + 1;

        // OppQualityIndex<Pixel8uC1> opp(iterations, repeats, imgWidth, imgHeight);
        //  TestOPPQualityIndex<Pixel8uC1> opp(iterations, repeats, imgWidth, imgHeight);
        //  TestNPPQualityIndex<nv::Image8uC1> npp(iterations, repeats, imgWidth, imgHeight);

        // opp.SetSrc(cpu_src1, cpu_src2);
        // npp.SetSrc(cpu_src1, cpu_src2);

        // opp.SetBorder(-1);
        // npp.SetBorder(-1);

        /* std::cout << "Warmup opp" << std::endl;
         opp.WarmUp();
         std::cout << "Running opp" << std::endl;
         Runtime rt_opp = opp.Run();
         std::cout << "Warmup npp" << std::endl;
         npp.WarmUp();
         std::cout << "Running npp" << std::endl;
         Runtime rt_npp = npp.Run();

         std::cout << "Done!" << std::endl << std::endl;
         std::cout << "OPP:" << std::endl;
         std::cout << rt_opp << std::endl;
         std::cout << "NPP:" << std::endl;
         std::cout << rt_npp << std::endl;

         const float ratio = rt_npp.Mean / rt_opp.Mean;
         if (ratio >= 1.0f)
         {
             std::cout << "OPP is " << ratio * 100.0f - 100.0f << "% faster!" << std::endl;
         }
         else
         {
             std::cout << "OPP is " << 1.0f / ratio * 100.0f - 100.0f << "% slower..." << std::endl;
         }*/

        // cpu::Image<Pixel32fC1> cpu_src1(imgWidth, imgHeight);
        //                        cpu::ImageView<Pixel8u3> cpu_src1A(cpu_src1);
        /*cpu::Image<Pixel8uC1> cpu_src1 =
            cpu::Image<Pixel8uC1>::Load("C:\\Users\\kunz_\\source\\repos\\oppV1\\opp\\test\\testData\\bird256bw.tif");
        cpu::Image<Pixel8uC1> cpu_src2 = cpu::Image<Pixel8uC1>::Load(
            "C:\\Users\\kunz_\\source\\repos\\oppV1\\opp\\test\\testData\\bird256bwnoisy.tif");
        cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load("F:\\ogpp\\muster.tif");*/
        // cpu::Image<Pixel32fC4> cpu_dst(imgWidth, imgHeight);
        // cpu::Image<Pixel32fC4> res_npp(imgWidth, imgHeight);
        // cpu::Image<Pixel32fC4> res_opp(imgWidth, imgHeight);

        // Image<Pixel32fC1> opp_src1(imgWidth, imgHeight);
        // Image<Pixel32fC1> opp_dst(imgWidth, imgHeight);

        // nv::Image8uC4 npp_src1(imgWidth, imgHeight);
        //   nv::Image8uC1 npp_dst(imgWidth, imgHeight);

        // cpu_dst.AverageError(cpu_src1, maxError, scalar);

        // npp_src1.SizeRoi();
        //  cpu_src1.Set(0);
        //  cpu_src1(7, 7) = 1;

        /*cpu_src1.SetRoi(Roi(116, 65, sizeTpl, sizeTpl));
        cpu_src1.Copy(cpu_tpl);
        cpu_src1.ResetRoi();*/

        // constexpr int filterSize = 11;
        //  DevVar<float> filter(to_size_t(filterSize));
        //  DevVar<float> filter2(64 * 64);
        //  std::vector<float> filter_h(to_size_t(filterSize));
        //  std::vector<float> filter2_h(64 * 64, 1.0f / (9.0f));
        //  std::vector<Pixel8uC1> mask_h   = {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        //  0}; std::vector<byte> mask_bh       = {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
        //  0, 0}; std::vector<Pixel32sC1> maski_h = {0,  0,  10, 0,   0,   0,   10, 10, 10, 0,   10, 10, 10,
        //                                     10, 10, 0,  -10, -10, -10, 0,  0,  0,  -10, 0,  0};
        //  std::vector<int> maski_bh       = {0,  0,  -10, 0,  0,  0,  -10, -10, -10, 0,  10, 10, 10,
        //                                     10, 10, 0,   10, 10, 10, 0,   0,   0,   10, 0,  0};
        //  DevVar<Pixel8uC1> mask(to_size_t(filterSize * filterSize));
        //  DevVar<byte> maskb(to_size_t(filterSize * filterSize));
        //  DevVar<Pixel32sC1> maski(to_size_t(filterSize * filterSize));
        //  DevVar<int> maskib(to_size_t(filterSize * filterSize));
        //  DevVar<double> meanTpl(1);
        //  mask << mask_h;
        //  maskb << mask_bh;
        //  maski << maski_h;
        //  maskib << maski_bh;
        //  constexpr double sigma = 1.5; // 0.4 + (to_double(filterSize) / 3.0) * 0.6

        // float sum_filter = 0;
        // for (size_t i = 0; i < to_size_t(filterSize); i++)
        //{
        //     double x    = to_double(i) - to_double(filterSize / 2);
        //     filter_h[i] = to_float(1.0 / (std::sqrt(2.0 * std::numbers::pi_v<double>) * sigma) *
        //                            std::exp(-(x * x) / (2.0 * sigma * sigma)));
        //     sum_filter += filter_h[i];
        // }
        // for (size_t i = 0; i < to_size_t(filterSize); i++)
        //{
        //     filter_h[i] /= sum_filter;
        // }

        // filter << filter_h;
        // filter2 << filter2_h;

        /*for (auto &pixel : cpu_src1)
        {
            pixel.Value().x = to_float(pixel.Pixel().x % 256);
        } */

        // cpu_src1.Set(1);
        //  cpu_src1.FillRandom(0);
        //   cpu_src2.FillRandom(1);
        //   cpu_src2.Div(4).Add(cpu_src1).Sub(2); /**/
        /*cpu_mask.FillRandom(2);
        cpu_mask.Div(127).Mul(255);*/
        /*cpu_src1.Set(1);
        cpu_mask.Set(1);
        cpu_src1.Mul(0.05f);
        cpu_src1.Add(cpu_src2);*/

        // NppStreamContext nppCtx;
        // NppStatus status = nppGetStreamContext(&nppCtx);
        // if (status != NPP_SUCCESS)
        //{
        //     return -1;
        // }

        // StreamCtx oppCtx = StreamCtxSingleton::Get();

        // cpu_src1 >> npp_src1;

        // cpu_src1 >> opp_src1;

        //  cpu_mask >> opp_mask;
        //  cpu_mask >> npp_mask;

        /*DevVar<byte> bufferMean(opp_tpl.MeanBufferSize(oppCtx));*/
        // DevVar<byte> bufferSSIM(opp_src1.SqrIntegralBufferSize(opp_dstSum, opp_dstSqr, oppCtx));

        // opp_src1.SetRoi(Border(-1, 0));
        // opp_dstAngle.Set(Pixel32fC1(0.0f), oppCtx);
        // npp_dstAngle.Set(Pixel32fC1(0.0f), nppCtx);
        /*opp_src1.SetRoi(-8);
        opp_dstAngle.SetRoi(-8);

        // npp_src1.SetRoi(-8);
        npp_dstAngle.SetRoi(-8); */
        // const BorderType opp_boder = BorderType::Replicate;
        /*const NppiMaskSize nppMask  = NppiMaskSize::NPP_MASK_SIZE_3_X_3;
        const MaskSize oppMask      = MaskSize::Mask_3x3;
        const FixedFilter oppFilter = FixedFilter::Laplace;*/
        // NppiSize nppMaskSize    = {filterSize, filterSize};
        // NppiPoint nppMaskCenter = {filterSize / 2, filterSize / 2};
        // Vec2i oppMaskSize   = {filterSize, filterSize};
        // Vec2i oppMaskCenter = {filterSize / 2, filterSize / 2};
        // DevVar<float> preComputedDist(64 * 64);
        // DevVar<float> preComputedDist2(64 * 64);
        // DevVar<Pixel32fC3> ssimo(1);
        // ssimo.Memset(0);
        // DevVar<float> ssim(1);
        // DevVar<float> mssim(3);
        // DevVar<float> wmssim(1);
        // DevVar<Pixel64fC1> qio(1);
        // DevVar<float> qin(1);

        // std::vector<float> preComputedH(64 * 64);
        // std::vector<float> preComputedH2(64 * 64);
        // const int maskSizeBilateral = 2;
        // const float posSquareSigma  = 15.0f;
        // for (int y = -maskSizeBilateral; y <= maskSizeBilateral; y++)
        //{
        //     const int idxY = y + maskSizeBilateral;
        //     for (int x = -maskSizeBilateral; x <= maskSizeBilateral; x++)
        //     {
        //         const int idxX = x + maskSizeBilateral;
        //         const int idx  = idxY * (maskSizeBilateral * 2 + 1) + idxX;

        //        const float fx = static_cast<float>(x);
        //        const float fy = static_cast<float>(y);
        //        float distSqr  = fx * fx + fy * fy;

        //        preComputedH[to_size_t(idx)] = std::exp(-distSqr / (2.0f * posSquareSigma));
        //        /*preComputedH[to_size_t(idx)] = 0;
        //        if (y == 0 && x == 0)
        //        {
        //            preComputedH[to_size_t(idx)] = 1;
        //        }*/
        //    }
        //}
        // preComputedDist << preComputedH;

        // float noise         = 100.0f;

        // npp_src1.FilterBorder32f(npp_dst, filter, {15, 15}, {6, 6}, NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);

        // opp_src1.ThresholdAdaptiveBoxFilter(opp_dst, oppMaskSize, oppMaskCenter, 0.5f, 255, 0, opp_boder, {0}, Roi(),
        //                                     oppCtx);
        // opp_src1.SeparableFilter(opp_dst, filter, oppMaskSize.x, oppMaskCenter.x, opp_boder, {0}, Roi(), oppCtx);
        // opp_src1.MaxFilter(opp_dst, oppMaskSize, opp_boder, {0}, Roi(), oppCtx);

        // opp_src1.GradientVectorSobel(opp_null16s, opp_null16s, opp_dstMag, opp_dstAngle, opp_null32fC4, Norm::L2,
        //                              MaskSize::Mask_5x5, opp_boder, {0}, Roi(), oppCtx);
        // opp_dstMag.CannyEdge(opp_dstAngle, opp_temp, opp_dst, 400, 3000, Roi(), oppCtx);
        // Pixel64fC1 ssimcpu;
        // cpu_src1.SSIM(cpu_src2, ssimcpu, 1);

        // opp_tpl2.Div(Pixel32fC1(255), oppCtx);
        // opp_src2.Div(Pixel32fC1(255), oppCtx);

        /// opp_src1.FixedFilter(opp_dst, FixedFilter::PrewittVert, MaskSize::Mask_3x3, BorderType::Replicate, oppCtx);
        /// cpu_src1.FixedFilter(cpu_dst, FixedFilter::PrewittVert, MaskSize::Mask_3x3, BorderType::Replicate);
        /// npp_src1.FilterPrewittVertBorder(npp_dst, NPP_BORDER_REPLICATE, nppCtx);

        // npp_src1.MSSSIM(npp_src2, mssim, bufferSSIM, nppCtx);
        // npp_src1.WMSSSIM(npp_src2, wmssim, bufferSSIM, nppCtx);
        // npp_src1.QualityIndex(npp_src2, {imgWidth, imgHeight}, qin, bufferSSIM, nppCtx);

        /*DevVarView<Pixel64fC1> tempvar(reinterpret_cast<Pixel64fC1 *>(meanTpl.Pointer()), 8);
        opp_tpl.Mean(tempvar, bufferMean, oppCtx);
        opp_src1.BoxAndSumSquareFilter(opp_boxFiltered, sizeTpl, BorderType::Constant, {0}, Roi(), oppCtx);
        opp_src1.CrossCorrelationCoefficient(opp_boxFiltered, opp_tpl, meanTpl, opp_dstAngle, BorderType::Constant, {0},
                                             Roi(), oppCtx);*/

        //  opp_src1.FixedFilter(opp_dst, oppFilter, oppMask, opp_boder, {0}, Roi(), oppCtx);
        //  npp_src1.FilterLaplaceBorder(npp_dst, nppMask, NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);
        /*npp_src1.FilterColumnBorder32f(npp_dst, filter, filterSize, filterSize / 2,
                                       NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);
        npp_src1.FilterBoxBorderAdvanced(npp_dst, nppMaskSize, nppMaskCenter, NppiBorderType::NPP_BORDER_REPLICATE,
                                         buffer, nppCtx);*/
        /*npp_src1.FilterMaxBorder(npp_dst, nppMaskSize, nppMaskCenter, NppiBorderType::NPP_BORDER_REPLICATE,
                                 nppCtx); */
        // npp_src1.Dilate3x3Border(npp_dst, NppiBorderType::NPP_BORDER_REPLICATE, nppCtx);

        // npp_src1.CrossCorrSame_NormLevelAdvanced(npp_tpl, npp_dstAngle, buffer1, buffer2, nppCtx);

        /*cpu_dst << opp_temp;
        cpu_dst.Save("f:\\cannyTempOpp.tif");
        cpu_dstMag << opp_dstMag;
        cpu_dstMag.Save("f:\\gradientVectorMagOpp.tif");
        cpu_dstAngle << opp_dstAngle;
        cpu_dstAngle.Save("f:\\gradientVectorAngleOpp.tif");

        nv::ImageView<Pixel8uC1> npp_temp0(reinterpret_cast<Pixel8uC1 *>(buffer.Pointer()),
                                           SizePitched({256, 256}, 256), npp_dst.ROI());
        nv::ImageView<Pixel16sC1> npp_temp1(reinterpret_cast<Pixel16sC1 *>(buffer.Pointer() + 256 * 256),
                                            SizePitched({256, 256}, 256 * 2), npp_dst.ROI());
        nv::ImageView<Pixel32fC1> npp_temp2(reinterpret_cast<Pixel32fC1 *>(buffer.Pointer() + 256 * 256 * 3),
                                            SizePitched({256, 256}, 256 * 4), npp_dst.ROI());
        nv::ImageView<Pixel16sC1> npp_temp3(reinterpret_cast<Pixel16sC1 *>(buffer.Pointer() + 256 * 256 * 7),
                                            SizePitched({256, 256}, 256 * 2), npp_dst.ROI());*/

        /*cpu_dst << npp_temp0;
        cpu_dst.Save("f:\\cannyTemp0Npp.tif");
        cpu_dstX << npp_temp1;
        cpu_dstX.Save("f:\\cannyTemp1Npp.tif");
        cpu_dstY << npp_temp3;
        cpu_dstY.Save("f:\\cannyTemp3Npp.tif");
        cpu_dstAngle << npp_dstAngle;
        cpu_dstAngle.Save("f:\\crossCorrNpp.tif");
        cpu_dstAngle << opp_dstAngle;
        cpu_dstAngle.Save("f:\\crossCorrOpp.tif");*/

        // cpu_src1.Save("f:\\lowpassOrig.tif");
        /*std::cout << std::endl << "Res CPU: " << std::endl;
        print(cpu_dst);
        cpu_dst.Save("f:\\filterCpu.tif");
        cpu_dst << opp_dst;
        std::cout << std::endl << "Res OPP: " << std::endl;
        print(cpu_dst);
        cpu_dst.Save("f:\\filterOpp.tif");
        cpu_dst << npp_dst;
        std::cout << std::endl << "Res NPP: " << std::endl;
        print(cpu_dst);
        cpu_dst.Save("f:\\filterNpp.tif"); */
    }
    catch (opp::OPPException &ex)
    {
        std::cout << ex.what();
    }
    catch (...)
    {
        return 1;
    }
    return 0;
}
