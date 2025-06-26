#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image64fC1View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size     = 256;
constexpr int size_tpl = 16;

TEST_CASE("8uC1", "[NPP.Statistics.CrossCorrelationCoefficient]")
{
    // const uint seed            = Catch::getSeed();
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load(root / "crossCorrTest.tif");
    cpu::Image<Pixel8uC1> cpu_tpl  = cpu::Image<Pixel8uC1>::Load(root / "template.tif");
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);

    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_tpl(size_tpl, size_tpl);
    nv::Image32fC1 npp_dst(size, size);
    size_t bufferSize =
        std::max(npp_dst.SameNormLevelGetBufferHostSize(nppCtx), npp_dst.ValidNormLevelGetBufferHostSize(nppCtx));
    mpp::cuda::DevVar<byte> buffer(bufferSize);

    cpu_src1 >> npp_src1;
    cpu_tpl >> npp_tpl;
    npp_dst.Set(0, nppCtx);
    npp_dst.SetRoi(Border(-8));
    npp_src1.CrossCorrValid_NormLevel(npp_tpl, npp_dst, buffer, nppCtx);
    npp_dst.ResetRoi();
    npp_dst.Threshold_GTVal(10, 0, nppCtx); // NPP sets the pixels with variance 0 to MAX_FLT
    npp_res << npp_dst;

    cpu_dst.Set(0);
    // we have to set roi slightly different compared to NPP in order to get exact same output:
    cpu_dst.SetRoi(Border(-8, -8, -7, -7));
    cpu_src1.SetRoi(Border(-8, -8, -7, -7));
    cpu_src1.CrossCorrelationCoefficient(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    cpu_dst.ResetRoi();
    cpu_src1.ResetRoi();
    npp_dst.ResetRoi();

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    npp_dst.Set(0, nppCtx);
    npp_src1.CrossCorrSame_NormLevel(npp_tpl, npp_dst, buffer, nppCtx);
    npp_dst.ResetRoi();
    npp_dst.Threshold_GTVal(10, 0, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set(0);
    cpu_src1.CrossCorrelationCoefficient(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("16uC1", "[NPP.Statistics.CrossCorrelationCoefficient]")
{
    // const uint seed            = Catch::getSeed();
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    cpu::Image<Pixel8uC1> img = cpu::Image<Pixel8uC1>::Load(root / "crossCorrTest.tif");
    cpu::Image<Pixel8uC1> tpl = cpu::Image<Pixel8uC1>::Load(root / "template.tif");

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_tpl(size_tpl, size_tpl);

    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);

    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_tpl(size_tpl, size_tpl);
    nv::Image32fC1 npp_dst(size, size);
    size_t bufferSize =
        std::max(npp_dst.SameNormLevelGetBufferHostSize(nppCtx), npp_dst.ValidNormLevelGetBufferHostSize(nppCtx));
    mpp::cuda::DevVar<byte> buffer(bufferSize);

    img.Convert(cpu_src1);
    tpl.Convert(cpu_tpl);

    cpu_src1 >> npp_src1;
    cpu_tpl >> npp_tpl;
    npp_dst.Set(0, nppCtx);
    npp_dst.SetRoi(Border(-8));
    npp_src1.CrossCorrValid_NormLevel(npp_tpl, npp_dst, buffer, nppCtx);
    npp_dst.ResetRoi();
    npp_dst.Threshold_GTVal(10, 0, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set(0);
    // we have to set roi slightly different compared to NPP in order to get exact same output:
    cpu_dst.SetRoi(Border(-8, -8, -7, -7));
    cpu_src1.SetRoi(Border(-8, -8, -7, -7));
    cpu_src1.CrossCorrelationCoefficient(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    cpu_dst.ResetRoi();
    cpu_src1.ResetRoi();
    npp_dst.ResetRoi();

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    npp_dst.Set(0, nppCtx);
    npp_src1.CrossCorrSame_NormLevel(npp_tpl, npp_dst, buffer, nppCtx);
    npp_dst.ResetRoi();
    npp_dst.Threshold_GTVal(10, 0, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set(0);
    cpu_src1.CrossCorrelationCoefficient(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("32fC1", "[NPP.Statistics.CrossCorrelationCoefficient]")
{
    // const uint seed            = Catch::getSeed();
    NppStreamContext nppCtx    = nv::Image8uC1::GetStreamContext();
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    cpu::Image<Pixel8uC1> img = cpu::Image<Pixel8uC1>::Load(root / "crossCorrTest.tif");
    cpu::Image<Pixel8uC1> tpl = cpu::Image<Pixel8uC1>::Load(root / "template.tif");

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_tpl(size_tpl, size_tpl);

    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);

    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_tpl(size_tpl, size_tpl);
    nv::Image32fC1 npp_dst(size, size);
    size_t bufferSize =
        std::max(npp_dst.SameNormLevelGetBufferHostSize(nppCtx), npp_dst.ValidNormLevelGetBufferHostSize(nppCtx));
    mpp::cuda::DevVar<byte> buffer(bufferSize);

    img.Convert(cpu_src1);
    tpl.Convert(cpu_tpl);

    cpu_src1 >> npp_src1;
    cpu_tpl >> npp_tpl;
    npp_dst.Set(0, nppCtx);
    npp_dst.SetRoi(Border(-8));
    npp_src1.CrossCorrValid_NormLevel(npp_tpl, npp_dst, buffer, nppCtx);
    npp_dst.ResetRoi();
    npp_dst.Threshold_GTVal(10, 0, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set(0);
    // we have to set roi slightly different compared to NPP in order to get exact same output:
    cpu_dst.SetRoi(Border(-8, -8, -7, -7));
    cpu_src1.SetRoi(Border(-8, -8, -7, -7));
    cpu_src1.CrossCorrelationCoefficient(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    cpu_dst.ResetRoi();
    cpu_src1.ResetRoi();
    npp_dst.ResetRoi();

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    npp_dst.Set(0, nppCtx);
    npp_src1.CrossCorrSame_NormLevel(npp_tpl, npp_dst, buffer, nppCtx);
    npp_dst.ResetRoi();
    npp_dst.Threshold_GTVal(10, 0, nppCtx);
    npp_res << npp_dst;

    cpu_dst.Set(0);
    cpu_src1.CrossCorrelationCoefficient(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}