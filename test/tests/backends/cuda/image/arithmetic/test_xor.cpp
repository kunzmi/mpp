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

TEST_CASE("8uC1", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Xor.UnevenSizeAlloc]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Xor.UnalignedPitch]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    gpu::ImageView<Pixel8uC1> gpu_src1Pitch(gpu_src1.Pointer(), {{size - 1, size - 1}, gpu_src1.Pitch() - 1});
    gpu::ImageView<Pixel8uC1> gpu_src2Pitch(gpu_src2.Pointer(), {{size - 1, size - 1}, gpu_src2.Pitch() - 1});
    gpu::ImageView<Pixel8uC1> gpu_dstPitch(gpu_dst.Pointer(), {{size - 1, size - 1}, gpu_dst.Pitch() - 1});

    CHECK_THROWS_AS(gpu_src1Pitch.Xor(gpu_src2Pitch, gpu_dstPitch), PitchException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Xor.NullPtr]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);

    gpu::ImageView<Pixel8uC1> gpu_dst(nullptr, {{size, size}, gpu_src1.Pitch()});

    CHECK_THROWS_AS(gpu_src1.Xor(gpu_src2, gpu_dst), NullPtrException);
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Xor.Roi]")
{
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_src2(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    gpu_src1.SetRoi(Border(-1));

    CHECK_THROWS_AS(gpu_src1.Xor(gpu_src2, gpu_dst), RoiException);
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel8sC1> cpu_src2(size, size);
    cpu::Image<Pixel8sC1> cpu_dst(size, size);
    cpu::Image<Pixel8sC1> gpu_res(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src2(size, size);
    gpu::Image<Pixel8sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Xor.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8sC1> cpu_src1(sizeUneven, sizeUneven);
    cpu::Image<Pixel8sC1> cpu_src2(sizeUneven, sizeUneven);
    cpu::Image<Pixel8sC1> cpu_dst(sizeUneven, sizeUneven);
    cpu::Image<Pixel8sC1> gpu_res(sizeUneven, sizeUneven);
    gpu::Image<Pixel8sC1> gpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8sC1> gpu_src2(sizeUneven, sizeUneven);
    gpu::Image<Pixel8sC1> gpu_dst(sizeUneven, sizeUneven);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Xor.UnalignedPitch]")
{
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src2(size, size);
    gpu::Image<Pixel8sC1> gpu_dst(size, size);

    gpu::ImageView<Pixel8sC1> gpu_src1Pitch(gpu_src1.Pointer(), {{size - 1, size - 1}, gpu_src1.Pitch() - 1});
    gpu::ImageView<Pixel8sC1> gpu_src2Pitch(gpu_src2.Pointer(), {{size - 1, size - 1}, gpu_src2.Pitch() - 1});
    gpu::ImageView<Pixel8sC1> gpu_dstPitch(gpu_dst.Pointer(), {{size - 1, size - 1}, gpu_dst.Pitch() - 1});

    CHECK_THROWS_AS(gpu_src1Pitch.Xor(gpu_src2Pitch, gpu_dstPitch), PitchException);
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Xor.NullPtr]")
{
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src2(size, size);

    gpu::ImageView<Pixel8sC1> gpu_dst(nullptr, {{size, size}, gpu_src1.Pitch()});

    CHECK_THROWS_AS(gpu_src1.Xor(gpu_src2, gpu_dst), NullPtrException);
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Xor.Roi]")
{
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_src2(size, size);
    gpu::Image<Pixel8sC1> gpu_dst(size, size);

    gpu_src1.SetRoi(Border(-1));

    CHECK_THROWS_AS(gpu_src1.Xor(gpu_src2, gpu_dst), RoiException);
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel8sC2> cpu_src2(size, size);
    cpu::Image<Pixel8sC2> cpu_dst(size, size);
    cpu::Image<Pixel8sC2> gpu_res(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_src2(size, size);
    gpu::Image<Pixel8sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel8sC3> cpu_src2(size, size);
    cpu::Image<Pixel8sC3> cpu_dst(size, size);
    cpu::Image<Pixel8sC3> gpu_res(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_src2(size, size);
    gpu::Image<Pixel8sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_src2(size, size);
    cpu::Image<Pixel8sC4> cpu_dst(size, size);
    cpu::Image<Pixel8sC4> gpu_res(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_src2(size, size);
    gpu::Image<Pixel8sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC4A", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    cpu::Image<Pixel8sC4A> cpu_src2(size, size);
    cpu::Image<Pixel8sC4A> cpu_dst(size, size);
    cpu::Image<Pixel8sC4A> gpu_res(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_src2(size, size);
    gpu::Image<Pixel8sC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_src2(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> gpu_res(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_src2(size, size);
    gpu::Image<Pixel16sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel16sC2> cpu_src2(size, size);
    cpu::Image<Pixel16sC2> cpu_dst(size, size);
    cpu::Image<Pixel16sC2> gpu_res(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_src2(size, size);
    gpu::Image<Pixel16sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_src2(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> gpu_res(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_src2(size, size);
    gpu::Image<Pixel16sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_src2(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> gpu_res(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_src2(size, size);
    gpu::Image<Pixel16sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC4A", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    cpu::Image<Pixel16sC4A> cpu_src2(size, size);
    cpu::Image<Pixel16sC4A> cpu_dst(size, size);
    cpu::Image<Pixel16sC4A> gpu_res(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_src2(size, size);
    gpu::Image<Pixel16sC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.Xor]")
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

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_src2(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_src2(size, size);
    gpu::Image<Pixel32sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_src2(size, size);
    cpu::Image<Pixel32sC4A> cpu_dst(size, size);
    cpu::Image<Pixel32sC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_src2(size, size);
    gpu::Image<Pixel32sC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC1", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel32uC1> cpu_src2(size, size);
    cpu::Image<Pixel32uC1> cpu_dst(size, size);
    cpu::Image<Pixel32uC1> gpu_res(size, size);
    gpu::Image<Pixel32uC1> gpu_src1(size, size);
    gpu::Image<Pixel32uC1> gpu_src2(size, size);
    gpu::Image<Pixel32uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC2", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC2> cpu_src1(size, size);
    cpu::Image<Pixel32uC2> cpu_src2(size, size);
    cpu::Image<Pixel32uC2> cpu_dst(size, size);
    cpu::Image<Pixel32uC2> gpu_res(size, size);
    gpu::Image<Pixel32uC2> gpu_src1(size, size);
    gpu::Image<Pixel32uC2> gpu_src2(size, size);
    gpu::Image<Pixel32uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC3", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC3> cpu_src1(size, size);
    cpu::Image<Pixel32uC3> cpu_src2(size, size);
    cpu::Image<Pixel32uC3> cpu_dst(size, size);
    cpu::Image<Pixel32uC3> gpu_res(size, size);
    gpu::Image<Pixel32uC3> gpu_src1(size, size);
    gpu::Image<Pixel32uC3> gpu_src2(size, size);
    gpu::Image<Pixel32uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC4", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4> cpu_src1(size, size);
    cpu::Image<Pixel32uC4> cpu_src2(size, size);
    cpu::Image<Pixel32uC4> cpu_dst(size, size);
    cpu::Image<Pixel32uC4> gpu_res(size, size);
    gpu::Image<Pixel32uC4> gpu_src1(size, size);
    gpu::Image<Pixel32uC4> gpu_src2(size, size);
    gpu::Image<Pixel32uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32uC4A", "[CUDA.Arithmetic.Xor]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32uC4A> cpu_src1(size, size);
    cpu::Image<Pixel32uC4A> cpu_src2(size, size);
    cpu::Image<Pixel32uC4A> cpu_dst(size, size);
    cpu::Image<Pixel32uC4A> gpu_res(size, size);
    gpu::Image<Pixel32uC4A> gpu_src1(size, size);
    gpu::Image<Pixel32uC4A> gpu_src2(size, size);
    gpu::Image<Pixel32uC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> gpu_src1;
    cpu_src2 >> gpu_src2;

    cpu_src1.Xor(cpu_src2, cpu_dst);
    gpu_src1.Xor(gpu_src2, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(cpu_src2);
    gpu_src1.Xor(gpu_src2);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.XorC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.XorC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC4> op(seed + 1);
    Pixel32sC4 constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.XorC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_dst(size, size);
    cpu::Image<Pixel32sC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC4A> op(seed + 1);
    Pixel32sC4A constVal;
    op(constVal);

    cpu_src1 >> gpu_src1;

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.XorDevC]")
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

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.XorDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC4> op(seed + 1);
    Pixel32sC4 constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32sC4> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.XorDevC]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_dst(size, size);
    cpu::Image<Pixel32sC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC4A> op(seed + 1);
    Pixel32sC4A constVal;
    op(constVal);
    mpp::cuda::DevVar<Pixel32sC4A> constDevVal(1);
    constDevVal << constVal;

    cpu_src1 >> gpu_src1;

    cpu_src1.Xor(constVal, cpu_dst);
    gpu_src1.Xor(constDevVal, gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Xor(constVal);
    gpu_src1.Xor(constDevVal);

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}
