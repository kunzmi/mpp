#include <backends/cuda/devVar.h>
#include <backends/cuda/image/image.h>
#include <backends/cuda/image/imageView.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/pitchException.h>
#include <common/image/roiException.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace gpu = mpp::image::cuda;

constexpr int size = 256;

TEST_CASE("8uC1", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -2);
    gpu_src1.Div(gpu_src2, gpu_dst, -2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -1);
    gpu_dst.Div(gpu_src2, -1);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Div.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8uC1> cpu_src1(sizeUneven, sizeUneven);
    cpu::Image<Pixel8uC1> cpu_src2(sizeUneven, sizeUneven);
    cpu::Image<Pixel8uC1> cpu_dst(sizeUneven, sizeUneven);
    cpu::Image<Pixel8uC1> gpu_res(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_src2(sizeUneven, sizeUneven);
    gpu::Image<Pixel8uC1> gpu_dst(sizeUneven, sizeUneven);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Div.EvenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeEven = 2 * size;
    cpu::Image<Pixel8uC1> cpu_src1(sizeEven, sizeEven);
    cpu::Image<Pixel8uC1> cpu_src2(sizeEven, sizeEven);
    cpu::Image<Pixel8uC1> cpu_dst(sizeEven, sizeEven);
    cpu::Image<Pixel8uC1> gpu_res(sizeEven, sizeEven);
    gpu::Image<Pixel8uC1> gpu_src1(sizeEven, sizeEven);
    gpu::Image<Pixel8uC1> gpu_src2(sizeEven, sizeEven);
    gpu::Image<Pixel8uC1> gpu_dst(sizeEven, sizeEven);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    for (int i = 0; i < 64; i++)
    {
        cpu_dst.Set(Pixel8uC1(0));
        gpu_dst.Set(Pixel8uC1(0));
        Roi roi = cpu_src1.ROI() - Border(1);

        cpu_src1.SetRoi(roi);
        cpu_src2.SetRoi(roi);
        cpu_dst.SetRoi(roi);
        gpu_src1.SetRoi(roi);
        gpu_src2.SetRoi(roi);
        gpu_dst.SetRoi(roi);
        cpu_src1.Div(cpu_src2, cpu_dst);
        gpu_src1.Div(gpu_src2, gpu_dst);

        cpu_dst.ResetRoi();
        gpu_dst.ResetRoi();
        gpu_res << gpu_dst;

        CHECK(cpu_dst.IsIdentical(gpu_res));
    }
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Div.UnalignedPitch]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    gpu::ImageView<Pixel8uC1> gpu_src1Pitch(gpu_src1.Pointer(), {{size - 1, size - 1}, gpu_src1.Pitch() - 1});
    gpu::ImageView<Pixel8uC1> gpu_src2Pitch(gpu_src2.Pointer(), {{size - 1, size - 1}, gpu_src2.Pitch() - 1});
    gpu::ImageView<Pixel8uC1> gpu_dstPitch(gpu_dst.Pointer(), {{size - 1, size - 1}, gpu_dst.Pitch() - 1});

    CHECK_THROWS_AS(gpu_src1Pitch.Div(gpu_src2Pitch, gpu_dstPitch), PitchException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Div.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);

    gpu::ImageView<Pixel8uC1> gpu_dst(nullptr, {{size, size}, gpu_src1.Pitch()});

    CHECK_THROWS_AS(gpu_src1.Div(gpu_src2, gpu_dst), NullPtrException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Div.Roi]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    gpu_src1.SetRoi(Border(-1));

    CHECK_THROWS_AS(gpu_src1.Div(gpu_src2, gpu_dst), RoiException);
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_src2(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_src2(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_src2(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_src2(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel8uC4A> cpu_src2(size, size);
    cpu::Image<Pixel8uC4A> cpu_dst(size, size);
    cpu::Image<Pixel8uC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_src2(size, size);
    gpu::Image<Pixel8uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, cpu_dst, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_src2(size, size);
    gpu::Image<Pixel16uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_src2(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_src2(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_src2(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_src2(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel16uC4A> cpu_src2(size, size);
    cpu::Image<Pixel16uC4A> cpu_dst(size, size);
    cpu::Image<Pixel16uC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_src2(size, size);
    gpu::Image<Pixel16uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, cpu_dst, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_src2(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_src2(size, size);
    gpu::Image<Pixel32sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_src2(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_src2(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_src2(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_src2(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_src2(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_src2(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_src2(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_src2(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_src2(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src2(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_src2(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_src2(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_src2(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_src2(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_src2(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_src2(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_src2(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_src2(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_src2(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(cpu_src2, cpu_dst);
    gpu_dst.Div(gpu_src2);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_src2(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_src2(size, size);
    gpu::Image<Pixel16scC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_src2(size, size);
    cpu::Image<Pixel16scC2> cpu_dst(size, size);
    cpu::Image<Pixel16scC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_src2(size, size);
    gpu::Image<Pixel16scC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_src2(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_src2(size, size);
    gpu::Image<Pixel16scC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.Div]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_src2(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_src2(size, size);
    gpu::Image<Pixel16scC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Div(cpu_src2, cpu_dst);
    gpu_src1.Div(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2, cpu_dst, -3);
    gpu_src1.Div(gpu_src2, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(cpu_src2, -8);
    gpu_dst.Div(gpu_src2, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(cpu_src2);
    gpu_src1.Div(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1.DivInv(cpu_src2);
    gpu_src1.DivInv(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC2> op(seed + 1);
    Pixel8uC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC3> op(seed + 1);
    Pixel8uC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel8uC4A> cpu_dst(size, size);
    cpu::Image<Pixel8uC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4A> op(seed + 1);
    Pixel8uC4A constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC2> op(seed + 1);
    Pixel16uC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC3> op(seed + 1);
    Pixel16uC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel16uC4A> cpu_dst(size, size);
    cpu::Image<Pixel16uC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4A> op(seed + 1);
    Pixel16uC4A constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC2> op(seed + 1);
    Pixel32sC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC3> op(seed + 1);
    Pixel32sC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC2> op(seed + 1);
    Pixel16fC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC3> op(seed + 1);
    Pixel16fC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC4> op(seed + 1);
    Pixel16fC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC2> op(seed + 1);
    Pixel32fC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC3> op(seed + 1);
    Pixel32fC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4A> op(seed + 1);
    Pixel32fC4A constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC1> op(seed + 1);
    Pixel16scC1 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_dst(size, size);
    cpu::Image<Pixel16scC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC2> op(seed + 1);
    Pixel16scC2 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC3> op(seed + 1);
    Pixel16scC3 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.DivC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC4> op(seed + 1);
    Pixel16scC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, gpu_dst, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC2> op(seed + 1);
    Pixel8uC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC3> op(seed + 1);
    Pixel8uC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4A> cpu_src1(size, size);
    cpu::Image<Pixel8uC4A> cpu_dst(size, size);
    cpu::Image<Pixel8uC4A> gpu_res(size, size);
    gpu::Image<Pixel8uC4A> gpu_src1(size, size);
    gpu::Image<Pixel8uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4A> op(seed + 1);
    Pixel8uC4A constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel8uC4A> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC2> op(seed + 1);
    Pixel16uC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC3> op(seed + 1);
    Pixel16uC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4A> cpu_src1(size, size);
    cpu::Image<Pixel16uC4A> cpu_dst(size, size);
    cpu::Image<Pixel16uC4A> gpu_res(size, size);
    gpu::Image<Pixel16uC4A> gpu_src1(size, size);
    gpu::Image<Pixel16uC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4A> op(seed + 1);
    Pixel16uC4A constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16uC4A> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32sC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC2> op(seed + 1);
    Pixel32sC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32sC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC3> op(seed + 1);
    Pixel32sC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32sC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16fC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC2> op(seed + 1);
    Pixel16fC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16fC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC3> op(seed + 1);
    Pixel16fC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16fC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC4> op(seed + 1);
    Pixel16fC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16fC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC2> op(seed + 1);
    Pixel32fC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC3> op(seed + 1);
    Pixel32fC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4A> cpu_src1(size, size);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    cpu::Image<Pixel32fC4A> gpu_res(size, size);
    gpu::Image<Pixel32fC4A> gpu_src1(size, size);
    gpu::Image<Pixel32fC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4A> op(seed + 1);
    Pixel32fC4A constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fC4A> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // InplaceInv
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC1> op(seed + 1);
    Pixel16scC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_dst(size, size);
    cpu::Image<Pixel16scC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC2> op(seed + 1);
    Pixel16scC2 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC2> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC3> op(seed + 1);
    Pixel16scC3 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC3> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC4> op(seed + 1);
    Pixel16scC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel16scC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    // the complex division is not exactly the same in floating point arithmetic,
    // so rounding leads to potential error of 1:
    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal, cpu_dst, -3);
    gpu_src1.Div(constDevVal, gpu_dst, -3);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    // Inplace
    cpu_dst.Div(constVal, -8);
    gpu_dst.Div(constDevVal, gpu_dst, -8);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 1));

    cpu_src1.Div(constVal);
    gpu_src1.Div(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 1));
}

TEST_CASE("32fcC1", "[CUDA.Arithmetic.DivDevC]")
{
    const uint seed = Catch ::getSeed();

    cpu::Image<Pixel32fcC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32fcC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fcC1> op(seed + 1);
    Pixel32fcC1 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32fcC1> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Div(constVal, cpu_dst);
    gpu_src1.Div(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_dst.Div(constVal);
    gpu_dst.Div(constDevVal);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // InplaceInv
    cpu_src1 >> gpu_src1; // make sure we have the same input
    cpu_src1.DivInv(constVal);
    gpu_src1.DivInv(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}