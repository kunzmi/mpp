#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/border.h>
#include <common/image/pitchException.h>
#include <common/image/pixelTypes.h>
#include <common/image/roiException.h>
#include <common/scratchBufferException.h>
#include <filesystem>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace gpu = mpp::image::cuda;

constexpr int size     = 256;
constexpr int size_tpl = 16;

TEST_CASE("8uC1", "[CUDA.Statistics.CrossCorrelationNormalized]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    cpu::Image<Pixel8uC1> cpu_src1 = cpu::Image<Pixel8uC1>::Load(root / "crossCorrTest.tif");
    cpu::Image<Pixel8uC1> cpu_tpl  = cpu::Image<Pixel8uC1>::Load(root / "template.tif");
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel8uC1> gpu_tpl(size_tpl, size_tpl);

    cpu_src1 >> gpu_src1;
    cpu_tpl >> gpu_tpl;
    gpu_dst.Set(Pixel32fC1(0));
    gpu_dst.SetRoi(Border(-8));
    gpu_src1.SetRoi(Border(-8));
    gpu_src1.CrossCorrelationNormalized(gpu_tpl, gpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    gpu_res << gpu_dst;

    cpu_dst.Set(0);
    cpu_dst.SetRoi(Border(-8));
    cpu_src1.SetRoi(Border(-8));
    cpu_src1.CrossCorrelationNormalized(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    cpu_dst.ResetRoi();
    gpu_dst.ResetRoi();

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Statistics.CrossCorrelationNormalized]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    cpu::Image<Pixel8uC1> img = cpu::Image<Pixel8uC1>::Load(root / "crossCorrTest.tif");
    cpu::Image<Pixel8uC1> tpl = cpu::Image<Pixel8uC1>::Load(root / "template.tif");

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_tpl(size_tpl, size_tpl);

    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);

    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel16uC1> gpu_tpl(size_tpl, size_tpl);

    img.Convert(cpu_src1);
    tpl.Convert(cpu_tpl);

    cpu_src1 >> gpu_src1;
    cpu_tpl >> gpu_tpl;
    gpu_dst.Set(Pixel32fC1(0));
    gpu_dst.SetRoi(Border(-8));
    gpu_src1.SetRoi(Border(-8));
    gpu_src1.CrossCorrelationNormalized(gpu_tpl, gpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    gpu_res << gpu_dst;

    cpu_dst.Set(0);
    cpu_dst.SetRoi(Border(-8));
    cpu_src1.SetRoi(Border(-8));
    cpu_src1.CrossCorrelationNormalized(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    cpu_dst.ResetRoi();
    gpu_dst.ResetRoi();

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Statistics.CrossCorrelationNormalized]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    cpu::Image<Pixel8uC1> img = cpu::Image<Pixel8uC1>::Load(root / "crossCorrTest.tif");
    cpu::Image<Pixel8uC1> tpl = cpu::Image<Pixel8uC1>::Load(root / "template.tif");

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_tpl(size_tpl, size_tpl);

    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);

    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);
    gpu::Image<Pixel32fC1> gpu_tpl(size_tpl, size_tpl);

    img.Convert(cpu_src1);
    tpl.Convert(cpu_tpl);

    cpu_src1 >> gpu_src1;
    cpu_tpl >> gpu_tpl;
    gpu_dst.Set(Pixel32fC1(0));
    gpu_dst.SetRoi(Border(-8));
    gpu_src1.SetRoi(Border(-8));
    gpu_src1.CrossCorrelationNormalized(gpu_tpl, gpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    gpu_res << gpu_dst;

    cpu_dst.Set(0);
    cpu_dst.SetRoi(Border(-8));
    cpu_src1.SetRoi(Border(-8));
    cpu_src1.CrossCorrelationNormalized(cpu_tpl, cpu_dst, {0}, BorderType::Constant, Roi(0, 0, size, size));
    cpu_dst.ResetRoi();
    gpu_dst.ResetRoi();

    CHECK(cpu_dst.IsIdentical(gpu_res));
}