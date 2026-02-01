#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC2View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/colorConversion/colorMatrices.h>
#include <common/defines.h>
#include <fstream>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

TEST_CASE("8uC4", "[NPP.ColorConversion.LUT3D]")
{
    std::filesystem::path root  = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC4> lut3d = cpu::Image<Pixel8uC4>::Load(root / "lut3d.tif");

    mpp::cuda::DevVar<Pixel8uC4> dlut3d(lut3d.SizeAlloc().TotalSize()); // 32*32*32 = 256*128
    dlut3d << lut3d.Pointer();

    std::vector<byte> levels(32);
    for (size_t i = 0; i < 32; i++)
    {
        levels[i] = static_cast<byte>(static_cast<float>(i) / 31.0f * 255.0f + 0.5f);
    }
    int aLevels[3]   = {32, 32, 32};
    byte *plevels[3] = {levels.data(), levels.data(), levels.data()};

    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    Size2D sizeFull                = cpu_src1.SizeAlloc();
    cpu::Image<Pixel8uC4> cpu_src2(sizeFull);
    cpu::Image<Pixel8uC3> cpu_src3(sizeFull);
    cpu::Image<Pixel8uC4> cpu_dst(sizeFull);
    cpu_src1.SwapChannel(cpu_src2, {0, 1, 2, 3}, 255);
    cpu_dst.Set({1, 2, 3, 4});

    nv::Image8uC4 npp_src1(sizeFull);
    nv::Image8uC4 npp_dst1(sizeFull);
    cpu_src2 >> npp_src1;
    cpu_dst >> npp_dst1;

    npp_src1.LUT_Trilinear(npp_dst1, dlut3d, plevels, aLevels, nppCtx);

    npp_dst1 >> cpu_dst;

    cpu_src1.LUTTrilinear(reinterpret_cast<Pixel8uC4A *>(lut3d.Pointer()), {0}, {255}, {32});

    cpu_dst.SwapChannel(cpu_src3, {0, 1, 2});

    Vec3d error;
    double err;
    cpu_src1.MaximumError(cpu_src3, error, err);

    CHECK(cpu_src1.IsSimilar(cpu_src3, 3));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.LUT]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    std::vector<int> lut1 = {36, 39, 51, 63, 65, 65, 66, 66, 90, 121, 130, 140, 158, 179, 192, 208, 215, 228, 237, 245};
    std::vector<int> lut2 = {4, 14, 20, 34, 73, 86, 90, 98, 120, 136, 141, 145, 146, 150, 193, 194, 199, 212, 234, 239};
    std::vector<int> lut3 = {20,  22,  39,  42,  43,  59,  68,  80,  113, 115,
                             135, 138, 154, 167, 176, 191, 203, 211, 233, 255};
    std::vector<Pixel8uC1> palette1(256);
    std::vector<Pixel8uC1> palette2(256);
    std::vector<Pixel8uC1> palette3(256);

    Pixel8uC1 *palettes[3] = {palette1.data(), palette2.data(), palette3.data()};

    std::vector<int> levels(lut1.size());
    for (size_t i = 0; i < lut1.size(); i++)
    {
        levels[i] = static_cast<int>(static_cast<float>(i) / 19.0f * 255.0f + 0.5f);
    }

    cuda::DevVar<int> d_lut1(lut1.size());
    cuda::DevVar<int> d_lut2(lut2.size());
    cuda::DevVar<int> d_lut3(lut3.size());
    cuda::DevVar<int> d_levels(lut3.size());

    d_lut1 << lut1;
    d_lut2 << lut2;
    d_lut3 << lut3;
    d_levels << levels;

    int aLevels[3] = {static_cast<int>(lut1.size()), static_cast<int>(lut2.size()), static_cast<int>(lut3.size())};
    cuda::DevVarView<int> plevels[3] = {d_levels, d_levels, d_levels};
    cuda::DevVarView<int> pluts[3]   = {d_lut1, d_lut2, d_lut3};

    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    Size2D sizeFull               = cpu_src.SizeAlloc();
    cpu::Image<Pixel8uC3> cpu_dst(sizeFull);
    cpu::Image<Pixel8uC3> npp_res(sizeFull);

    nv::Image8uC3 npp_src(sizeFull);
    nv::Image8uC3 npp_dst(sizeFull);
    cpu_src >> npp_src;
    cpu_dst >> npp_dst;

    npp_src.LUT(npp_dst, pluts, plevels, aLevels, nppCtx);
    npp_dst >> npp_res;

    cpu_src.LUTToPalette(levels.data(), lut1.data(), static_cast<int>(lut1.size()), palette1.data(),
                         InterpolationMode::NearestNeighbor);
    cpu_src.LUTToPalette(levels.data(), lut2.data(), static_cast<int>(lut2.size()), palette2.data(),
                         InterpolationMode::NearestNeighbor);
    cpu_src.LUTToPalette(levels.data(), lut3.data(), static_cast<int>(lut3.size()), palette3.data(),
                         InterpolationMode::NearestNeighbor);

    cpu_src.LUTPalette(cpu_dst, palettes, 8);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.LUTLinear]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    std::vector<int> lut1 = {36, 39, 51, 63, 65, 65, 66, 66, 90, 121, 130, 140, 158, 179, 192, 208, 215, 228, 237, 245};
    std::vector<int> lut2 = {4, 14, 20, 34, 73, 86, 90, 98, 120, 136, 141, 145, 146, 150, 193, 194, 199, 212, 234, 239};
    std::vector<int> lut3 = {20,  22,  39,  42,  43,  59,  68,  80,  113, 115,
                             135, 138, 154, 167, 176, 191, 203, 211, 233, 255};
    std::vector<Pixel8uC1> palette1(256);
    std::vector<Pixel8uC1> palette2(256);
    std::vector<Pixel8uC1> palette3(256);

    Pixel8uC1 *palettes[3] = {palette1.data(), palette2.data(), palette3.data()};

    std::vector<int> levels(lut1.size());
    for (size_t i = 0; i < lut1.size(); i++)
    {
        levels[i] = static_cast<int>(static_cast<float>(i) / 19.0f * 255.0f + 0.5f);
    }

    cuda::DevVar<int> d_lut1(lut1.size());
    cuda::DevVar<int> d_lut2(lut2.size());
    cuda::DevVar<int> d_lut3(lut3.size());
    cuda::DevVar<int> d_levels(lut3.size());

    d_lut1 << lut1;
    d_lut2 << lut2;
    d_lut3 << lut3;
    d_levels << levels;

    int aLevels[3] = {static_cast<int>(lut1.size()), static_cast<int>(lut2.size()), static_cast<int>(lut3.size())};
    cuda::DevVarView<int> plevels[3] = {d_levels, d_levels, d_levels};
    cuda::DevVarView<int> pluts[3]   = {d_lut1, d_lut2, d_lut3};

    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    Size2D sizeFull               = cpu_src.SizeAlloc();
    cpu::Image<Pixel8uC3> cpu_dst(sizeFull);
    cpu::Image<Pixel8uC3> npp_res(sizeFull);

    nv::Image8uC3 npp_src(sizeFull);
    nv::Image8uC3 npp_dst(sizeFull);
    cpu_src >> npp_src;
    cpu_dst >> npp_dst;

    npp_src.LUT_Linear(npp_dst, pluts, plevels, aLevels, nppCtx);
    npp_dst >> npp_res;

    cpu_src.LUTToPalette(levels.data(), lut1.data(), static_cast<int>(lut1.size()), palette1.data(),
                         InterpolationMode::Linear);
    cpu_src.LUTToPalette(levels.data(), lut2.data(), static_cast<int>(lut2.size()), palette2.data(),
                         InterpolationMode::Linear);
    cpu_src.LUTToPalette(levels.data(), lut3.data(), static_cast<int>(lut3.size()), palette3.data(),
                         InterpolationMode::Linear);

    cpu_src.LUTPalette(cpu_dst, palettes, 8);

    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC3", "[NPP.ColorConversion.LUTCubic]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    std::vector<int> lut1 = {36, 39, 51, 63, 65, 65, 66, 66, 90, 121, 130, 140, 158, 179, 192, 208, 215, 228, 237, 245};
    std::vector<int> lut2 = {4, 14, 20, 34, 73, 86, 90, 98, 120, 136, 141, 145, 146, 150, 193, 194, 199, 212, 234, 239};
    std::vector<int> lut3 = {20,  22,  39,  42,  43,  59,  68,  80,  113, 115,
                             135, 138, 154, 167, 176, 191, 203, 211, 233, 255};
    std::vector<Pixel8uC1> palette1(256);
    std::vector<Pixel8uC1> palette2(256);
    std::vector<Pixel8uC1> palette3(256);

    Pixel8uC1 *palettes[3] = {palette1.data(), palette2.data(), palette3.data()};

    std::vector<int> levels(lut1.size());
    for (size_t i = 0; i < lut1.size(); i++)
    {
        levels[i] = static_cast<int>(static_cast<float>(i) / 19.0f * 255.0f + 0.5f);
    }

    cuda::DevVar<int> d_lut1(lut1.size());
    cuda::DevVar<int> d_lut2(lut2.size());
    cuda::DevVar<int> d_lut3(lut3.size());
    cuda::DevVar<int> d_levels(lut3.size());

    d_lut1 << lut1;
    d_lut2 << lut2;
    d_lut3 << lut3;
    d_levels << levels;

    int aLevels[3] = {static_cast<int>(lut1.size()), static_cast<int>(lut2.size()), static_cast<int>(lut3.size())};
    cuda::DevVarView<int> plevels[3] = {d_levels, d_levels, d_levels};
    cuda::DevVarView<int> pluts[3]   = {d_lut1, d_lut2, d_lut3};

    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    Size2D sizeFull               = cpu_src.SizeAlloc();
    cpu::Image<Pixel8uC3> cpu_dst(sizeFull);
    cpu::Image<Pixel8uC3> npp_res(sizeFull);

    nv::Image8uC3 npp_src(sizeFull);
    nv::Image8uC3 npp_dst(sizeFull);
    cpu_src >> npp_src;
    cpu_dst >> npp_dst;

    npp_src.LUT_Cubic(npp_dst, pluts, plevels, aLevels, nppCtx);
    npp_dst >> npp_res;

    cpu_src.LUTToPalette(levels.data(), lut1.data(), static_cast<int>(lut1.size()), palette1.data(),
                         InterpolationMode::CubicLagrange);
    cpu_src.LUTToPalette(levels.data(), lut2.data(), static_cast<int>(lut2.size()), palette2.data(),
                         InterpolationMode::CubicLagrange);
    cpu_src.LUTToPalette(levels.data(), lut3.data(), static_cast<int>(lut3.size()), palette3.data(),
                         InterpolationMode::CubicLagrange);

    cpu_src.LUTPalette(cpu_dst, palettes, 8);

    // NPP's cubic interpolation does not fit to the intepolation as done in MPP and IPP
    CHECK(cpu_dst.IsSimilar(npp_res, 3));
}

TEST_CASE("32fC3", "[NPP.ColorConversion.LUT]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    std::vector<float> lut1 = {36,  39,  51,  63,  65,  65,  66,  66,  90,  121,
                               130, 140, 158, 179, 192, 208, 215, 228, 237, 245};
    std::vector<float> lut2 = {4,   14,  20,  34,  73,  86,  90,  98,  120, 136,
                               141, 145, 146, 150, 193, 194, 199, 212, 234, 239};
    std::vector<float> lut3 = {20,  22,  39,  42,  43,  59,  68,  80,  113, 115,
                               135, 138, 154, 167, 176, 191, 203, 211, 233, 255};

    Pixel32fC1 *values[3] = {reinterpret_cast<Pixel32fC1 *>(lut1.data()), reinterpret_cast<Pixel32fC1 *>(lut2.data()),
                             reinterpret_cast<Pixel32fC1 *>(lut3.data())};

    std::vector<float> levels(lut1.size());
    for (size_t i = 0; i < lut1.size(); i++)
    {
        levels[i] = static_cast<float>(i) / 19.0f * 255.0f;
    }

    cuda::DevVar<float> d_lut1(lut1.size());
    cuda::DevVar<float> d_lut2(lut2.size());
    cuda::DevVar<float> d_lut3(lut3.size());
    cuda::DevVar<float> d_levels(lut3.size());

    d_lut1 << lut1;
    d_lut2 << lut2;
    d_lut3 << lut3;
    d_levels << levels;

    int aLevels[3] = {static_cast<int>(lut1.size()), static_cast<int>(lut2.size()), static_cast<int>(lut3.size())};
    cuda::DevVarView<float> plevels[3] = {d_levels, d_levels, d_levels};
    cuda::DevVarView<float> pluts[3]   = {d_lut1, d_lut2, d_lut3};

    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    Size2D sizeFull                = cpu_src1.SizeAlloc();
    cpu::Image<Pixel32fC3> cpu_src(sizeFull);
    cpu::Image<Pixel32fC3> cpu_dst(sizeFull);
    cpu::Image<Pixel32fC3> npp_res(sizeFull);

    nv::Image32fC3 npp_src(sizeFull);
    nv::Image32fC3 npp_dst(sizeFull);
    cpu_src1.Convert(cpu_src);
    cpu_src >> npp_src;
    cpu_dst >> npp_dst;

    npp_src.LUT(npp_dst, pluts, plevels, aLevels, nppCtx);
    npp_dst >> npp_res;

    std::vector<int> accelerator(256);
    int *accelerators[3]        = {accelerator.data(), accelerator.data(), accelerator.data()};
    int acceleratorSizes[3]     = {256, 256, 256};
    Pixel32fC1 *levelsPixels[3] = {reinterpret_cast<Pixel32fC1 *>(levels.data()),
                                   reinterpret_cast<Pixel32fC1 *>(levels.data()),
                                   reinterpret_cast<Pixel32fC1 *>(levels.data())};

    cpu_src.LUTAccelerator(reinterpret_cast<Pixel32fC1 *>(levels.data()), accelerator.data(),
                           static_cast<int>(lut1.size()), static_cast<int>(accelerator.size()));

    cpu_src.LUT(cpu_dst, levelsPixels, values, accelerators, aLevels, acceleratorSizes,
                InterpolationMode::NearestNeighbor);

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.ColorConversion.LUTLinear]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    std::vector<float> lut1 = {36,  39,  51,  63,  65,  65,  66,  66,  90,  121,
                               130, 140, 158, 179, 192, 208, 215, 228, 237, 245};
    std::vector<float> lut2 = {4,   14,  20,  34,  73,  86,  90,  98,  120, 136,
                               141, 145, 146, 150, 193, 194, 199, 212, 234, 239};
    std::vector<float> lut3 = {20,  22,  39,  42,  43,  59,  68,  80,  113, 115,
                               135, 138, 154, 167, 176, 191, 203, 211, 233, 255};

    Pixel32fC1 *values[3] = {reinterpret_cast<Pixel32fC1 *>(lut1.data()), reinterpret_cast<Pixel32fC1 *>(lut2.data()),
                             reinterpret_cast<Pixel32fC1 *>(lut3.data())};

    std::vector<float> levels(lut1.size());
    for (size_t i = 0; i < lut1.size(); i++)
    {
        levels[i] = static_cast<float>(i) / 19.0f * 255.0f;
    }

    cuda::DevVar<float> d_lut1(lut1.size());
    cuda::DevVar<float> d_lut2(lut2.size());
    cuda::DevVar<float> d_lut3(lut3.size());
    cuda::DevVar<float> d_levels(lut3.size());

    d_lut1 << lut1;
    d_lut2 << lut2;
    d_lut3 << lut3;
    d_levels << levels;

    int aLevels[3] = {static_cast<int>(lut1.size()), static_cast<int>(lut2.size()), static_cast<int>(lut3.size())};
    cuda::DevVarView<float> plevels[3] = {d_levels, d_levels, d_levels};
    cuda::DevVarView<float> pluts[3]   = {d_lut1, d_lut2, d_lut3};

    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    Size2D sizeFull                = cpu_src1.SizeAlloc();
    cpu::Image<Pixel32fC3> cpu_src(sizeFull);
    cpu::Image<Pixel32fC3> cpu_dst(sizeFull);
    cpu::Image<Pixel32fC3> npp_res(sizeFull);

    nv::Image32fC3 npp_src(sizeFull);
    nv::Image32fC3 npp_dst(sizeFull);
    cpu_src1.Convert(cpu_src);
    cpu_src >> npp_src;
    cpu_dst >> npp_dst;

    npp_src.LUT_Linear(npp_dst, pluts, plevels, aLevels, nppCtx);
    npp_dst >> npp_res;

    std::vector<int> accelerator(256);
    int *accelerators[3]        = {accelerator.data(), accelerator.data(), accelerator.data()};
    int acceleratorSizes[3]     = {256, 256, 256};
    Pixel32fC1 *levelsPixels[3] = {reinterpret_cast<Pixel32fC1 *>(levels.data()),
                                   reinterpret_cast<Pixel32fC1 *>(levels.data()),
                                   reinterpret_cast<Pixel32fC1 *>(levels.data())};

    cpu_src.LUTAccelerator(reinterpret_cast<Pixel32fC1 *>(levels.data()), accelerator.data(),
                           static_cast<int>(lut1.size()), static_cast<int>(accelerator.size()));

    cpu_src.LUT(cpu_dst, levelsPixels, values, accelerators, aLevels, acceleratorSizes, InterpolationMode::Linear);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.0001f));
}

TEST_CASE("32fC3", "[NPP.ColorConversion.LUTCubic]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    std::vector<float> lut1 = {36,  39,  51,  63,  65,  65,  66,  66,  90,  121,
                               130, 140, 158, 179, 192, 208, 215, 228, 237, 245};
    std::vector<float> lut2 = {4,   14,  20,  34,  73,  86,  90,  98,  120, 136,
                               141, 145, 146, 150, 193, 194, 199, 212, 234, 239};
    std::vector<float> lut3 = {20,  22,  39,  42,  43,  59,  68,  80,  113, 115,
                               135, 138, 154, 167, 176, 191, 203, 211, 233, 255};

    Pixel32fC1 *values[3] = {reinterpret_cast<Pixel32fC1 *>(lut1.data()), reinterpret_cast<Pixel32fC1 *>(lut2.data()),
                             reinterpret_cast<Pixel32fC1 *>(lut3.data())};

    std::vector<float> levels(lut1.size());
    for (size_t i = 0; i < lut1.size(); i++)
    {
        levels[i] = static_cast<float>(i) / 19.0f * 255.0f;
    }

    cuda::DevVar<float> d_lut1(lut1.size());
    cuda::DevVar<float> d_lut2(lut2.size());
    cuda::DevVar<float> d_lut3(lut3.size());
    cuda::DevVar<float> d_levels(lut3.size());

    d_lut1 << lut1;
    d_lut2 << lut2;
    d_lut3 << lut3;
    d_levels << levels;

    int aLevels[3] = {static_cast<int>(lut1.size()), static_cast<int>(lut2.size()), static_cast<int>(lut3.size())};
    cuda::DevVarView<float> plevels[3] = {d_levels, d_levels, d_levels};
    cuda::DevVarView<float> pluts[3]   = {d_lut1, d_lut2, d_lut3};

    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1 = cpu::Image<Pixel8uC3>::Load(root / "bird256.tif");
    Size2D sizeFull                = cpu_src1.SizeAlloc();
    cpu::Image<Pixel32fC3> cpu_src(sizeFull);
    cpu::Image<Pixel32fC3> cpu_dst(sizeFull);
    cpu::Image<Pixel32fC3> npp_res(sizeFull);

    nv::Image32fC3 npp_src(sizeFull);
    nv::Image32fC3 npp_dst(sizeFull);
    cpu_src1.Convert(cpu_src);
    cpu_src >> npp_src;
    cpu_dst >> npp_dst;

    npp_src.LUT_Cubic(npp_dst, pluts, plevels, aLevels, nppCtx);
    npp_dst >> npp_res;

    std::vector<int> accelerator(256);
    int *accelerators[3]        = {accelerator.data(), accelerator.data(), accelerator.data()};
    int acceleratorSizes[3]     = {256, 256, 256};
    Pixel32fC1 *levelsPixels[3] = {reinterpret_cast<Pixel32fC1 *>(levels.data()),
                                   reinterpret_cast<Pixel32fC1 *>(levels.data()),
                                   reinterpret_cast<Pixel32fC1 *>(levels.data())};

    cpu_src.LUTAccelerator(reinterpret_cast<Pixel32fC1 *>(levels.data()), accelerator.data(),
                           static_cast<int>(lut1.size()), static_cast<int>(accelerator.size()));

    cpu_src.LUT(cpu_dst, levelsPixels, values, accelerators, aLevels, acceleratorSizes,
                InterpolationMode::CubicLagrange);

    // again, the Cubic interpolation in NPP is off compared to IPP and MPP
    CHECK(cpu_dst.IsSimilar(npp_res, 3.0f));
}
