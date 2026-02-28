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

TEST_CASE("8sC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    cpu::Image<Pixel8sC1> cpu_dst(size, size);
    cpu::Image<Pixel8sC1> gpu_res(size, size);
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Exp.UnevenSizeAlloc]")
{
    const uint seed = Catch::getSeed();

    constexpr int sizeUneven = 2 * size - 1;
    cpu::Image<Pixel8sC1> cpu_src1(sizeUneven, sizeUneven);
    cpu::Image<Pixel8sC1> cpu_dst(sizeUneven, sizeUneven);
    cpu::Image<Pixel8sC1> gpu_res(sizeUneven, sizeUneven);
    gpu::Image<Pixel8sC1> gpu_src1(sizeUneven, sizeUneven);
    gpu::Image<Pixel8sC1> gpu_dst(sizeUneven, sizeUneven);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Exp.UnalignedPitch]")
{
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_dst(size, size);

    gpu::ImageView<Pixel8sC1> gpu_src1Pitch(gpu_src1.Pointer(), {{size - 1, size - 1}, gpu_src1.Pitch() - 1});
    gpu::ImageView<Pixel8sC1> gpu_dstPitch(gpu_dst.Pointer(), {{size - 1, size - 1}, gpu_dst.Pitch() - 1});

    CHECK_THROWS_AS(gpu_src1Pitch.Exp(gpu_dstPitch), PitchException);
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Exp.NullPtr]")
{
    gpu::Image<Pixel8sC1> gpu_src1(size, size);

    gpu::ImageView<Pixel8sC1> gpu_dst(nullptr, {{size, size}, gpu_src1.Pitch()});

    CHECK_THROWS_AS(gpu_src1.Exp(gpu_dst), NullPtrException);
}

TEST_CASE("8sC1", "[CUDA.Arithmetic.Exp.Roi]")
{
    gpu::Image<Pixel8sC1> gpu_src1(size, size);
    gpu::Image<Pixel8sC1> gpu_dst(size, size);

    gpu_src1.SetRoi(Border(-1));

    CHECK_THROWS_AS(gpu_src1.Exp(gpu_dst), RoiException);
}

TEST_CASE("8sC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC2> cpu_src1(size, size);
    cpu::Image<Pixel8sC2> cpu_dst(size, size);
    cpu::Image<Pixel8sC2> gpu_res(size, size);
    gpu::Image<Pixel8sC2> gpu_src1(size, size);
    gpu::Image<Pixel8sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    cpu::Image<Pixel8sC3> cpu_dst(size, size);
    cpu::Image<Pixel8sC3> gpu_res(size, size);
    gpu::Image<Pixel8sC3> gpu_src1(size, size);
    gpu::Image<Pixel8sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4> cpu_src1(size, size);
    cpu::Image<Pixel8sC4> cpu_dst(size, size);
    cpu::Image<Pixel8sC4> gpu_res(size, size);
    gpu::Image<Pixel8sC4> gpu_src1(size, size);
    gpu::Image<Pixel8sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8sC4A", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8sC4A> cpu_src1(size, size);
    cpu::Image<Pixel8sC4A> cpu_dst(size, size);
    cpu::Image<Pixel8sC4A> gpu_res(size, size);
    gpu::Image<Pixel8sC4A> gpu_src1(size, size);
    gpu::Image<Pixel8sC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> gpu_res(size, size);
    gpu::Image<Pixel8uC1> gpu_src1(size, size);
    gpu::Image<Pixel8uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC2> cpu_src1(size, size);
    cpu::Image<Pixel8uC2> cpu_dst(size, size);
    cpu::Image<Pixel8uC2> gpu_res(size, size);
    gpu::Image<Pixel8uC2> gpu_src1(size, size);
    gpu::Image<Pixel8uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> gpu_res(size, size);
    gpu::Image<Pixel8uC3> gpu_src1(size, size);
    gpu::Image<Pixel8uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> gpu_res(size, size);
    gpu::Image<Pixel8uC4> gpu_src1(size, size);
    gpu::Image<Pixel8uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("8uC4A", "[CUDA.Arithmetic.Exp]")
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

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> gpu_res(size, size);
    gpu::Image<Pixel16sC1> gpu_src1(size, size);
    gpu::Image<Pixel16sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC2> cpu_src1(size, size);
    cpu::Image<Pixel16sC2> cpu_dst(size, size);
    cpu::Image<Pixel16sC2> gpu_res(size, size);
    gpu::Image<Pixel16sC2> gpu_src1(size, size);
    gpu::Image<Pixel16sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> gpu_res(size, size);
    gpu::Image<Pixel16sC3> gpu_src1(size, size);
    gpu::Image<Pixel16sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> gpu_res(size, size);
    gpu::Image<Pixel16sC4> gpu_src1(size, size);
    gpu::Image<Pixel16sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16sC4A", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16sC4A> cpu_src1(size, size);
    cpu::Image<Pixel16sC4A> cpu_dst(size, size);
    cpu::Image<Pixel16sC4A> gpu_res(size, size);
    gpu::Image<Pixel16sC4A> gpu_src1(size, size);
    gpu::Image<Pixel16sC4A> gpu_dst(size, size);

    cpu_dst.Set({127, 127, 127});
    gpu_dst.Set({127, 127, 127});

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> gpu_res(size, size);
    gpu::Image<Pixel16uC1> gpu_src1(size, size);
    gpu::Image<Pixel16uC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC2> cpu_src1(size, size);
    cpu::Image<Pixel16uC2> cpu_dst(size, size);
    cpu::Image<Pixel16uC2> gpu_res(size, size);
    gpu::Image<Pixel16uC2> gpu_src1(size, size);
    gpu::Image<Pixel16uC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> gpu_res(size, size);
    gpu::Image<Pixel16uC3> gpu_src1(size, size);
    gpu::Image<Pixel16uC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> gpu_res(size, size);
    gpu::Image<Pixel16uC4> gpu_src1(size, size);
    gpu::Image<Pixel16uC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16uC4A", "[CUDA.Arithmetic.Exp]")
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

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> gpu_res(size, size);
    gpu::Image<Pixel16scC1> gpu_src1(size, size);
    gpu::Image<Pixel16scC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC2> cpu_src1(size, size);
    cpu::Image<Pixel16scC2> cpu_dst(size, size);
    cpu::Image<Pixel16scC2> gpu_res(size, size);
    gpu::Image<Pixel16scC2> gpu_src1(size, size);
    gpu::Image<Pixel16scC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> gpu_res(size, size);
    gpu::Image<Pixel16scC3> gpu_src1(size, size);
    gpu::Image<Pixel16scC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16scC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> gpu_res(size, size);
    gpu::Image<Pixel16scC4> gpu_src1(size, size);
    gpu::Image<Pixel16scC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> gpu_res(size, size);
    gpu::Image<Pixel32sC1> gpu_src1(size, size);
    gpu::Image<Pixel32sC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC2> cpu_src1(size, size);
    cpu::Image<Pixel32sC2> cpu_dst(size, size);
    cpu::Image<Pixel32sC2> gpu_res(size, size);
    gpu::Image<Pixel32sC2> gpu_src1(size, size);
    gpu::Image<Pixel32sC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> gpu_res(size, size);
    gpu::Image<Pixel32sC3> gpu_src1(size, size);
    gpu::Image<Pixel32sC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4> cpu_src1(size, size);
    cpu::Image<Pixel32sC4> cpu_dst(size, size);
    cpu::Image<Pixel32sC4> gpu_res(size, size);
    gpu::Image<Pixel32sC4> gpu_src1(size, size);
    gpu::Image<Pixel32sC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32sC4A", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32sC4A> cpu_src1(size, size);
    cpu::Image<Pixel32sC4A> cpu_dst(size, size);
    cpu::Image<Pixel32sC4A> gpu_res(size, size);
    gpu::Image<Pixel32sC4A> gpu_src1(size, size);
    gpu::Image<Pixel32sC4A> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> gpu_res(size, size);
    gpu::Image<Pixel16fC1> gpu_src1(size, size);
    gpu::Image<Pixel16fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC2> cpu_src1(size, size);
    cpu::Image<Pixel16fC2> cpu_dst(size, size);
    cpu::Image<Pixel16fC2> gpu_res(size, size);
    gpu::Image<Pixel16fC2> gpu_src1(size, size);
    gpu::Image<Pixel16fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> gpu_res(size, size);
    gpu::Image<Pixel16fC3> gpu_src1(size, size);
    gpu::Image<Pixel16fC3> gpu_src2(size, size);
    gpu::Image<Pixel16fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("16fC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> gpu_res(size, size);
    gpu::Image<Pixel16fC4> gpu_src1(size, size);
    gpu::Image<Pixel16fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsIdentical(gpu_res));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsIdentical(gpu_res));
}

TEST_CASE("32fC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> gpu_res(size, size);
    gpu::Image<Pixel32fC1> gpu_src1(size, size);
    gpu::Image<Pixel32fC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC2> cpu_src1(size, size);
    cpu::Image<Pixel32fC2> cpu_dst(size, size);
    cpu::Image<Pixel32fC2> gpu_res(size, size);
    gpu::Image<Pixel32fC2> gpu_src1(size, size);
    gpu::Image<Pixel32fC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> gpu_res(size, size);
    gpu::Image<Pixel32fC3> gpu_src1(size, size);
    gpu::Image<Pixel32fC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> gpu_res(size, size);
    gpu::Image<Pixel32fC4> gpu_src1(size, size);
    gpu::Image<Pixel32fC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fC4A", "[CUDA.Arithmetic.Exp]")
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

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fcC1", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC1> cpu_src1(size, size);
    cpu::Image<Pixel32fcC1> cpu_dst(size, size);
    cpu::Image<Pixel32fcC1> gpu_res(size, size);
    gpu::Image<Pixel32fcC1> gpu_src1(size, size);
    gpu::Image<Pixel32fcC1> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fcC2", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC2> cpu_src1(size, size);
    cpu::Image<Pixel32fcC2> cpu_dst(size, size);
    cpu::Image<Pixel32fcC2> gpu_res(size, size);
    gpu::Image<Pixel32fcC2> gpu_src1(size, size);
    gpu::Image<Pixel32fcC2> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fcC3", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC3> cpu_src1(size, size);
    cpu::Image<Pixel32fcC3> cpu_dst(size, size);
    cpu::Image<Pixel32fcC3> gpu_res(size, size);
    gpu::Image<Pixel32fcC3> gpu_src1(size, size);
    gpu::Image<Pixel32fcC3> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}

TEST_CASE("32fcC4", "[CUDA.Arithmetic.Exp]")
{
    const uint seed = Catch::getSeed();

    cpu::Image<Pixel32fcC4> cpu_src1(size, size);
    cpu::Image<Pixel32fcC4> cpu_dst(size, size);
    cpu::Image<Pixel32fcC4> gpu_res(size, size);
    gpu::Image<Pixel32fcC4> gpu_src1(size, size);
    gpu::Image<Pixel32fcC4> gpu_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> gpu_src1;

    cpu_src1.Exp(cpu_dst);
    gpu_src1.Exp(gpu_dst);

    gpu_res << gpu_dst;

    CHECK(cpu_dst.IsSimilar(gpu_res, 0.0001f));

    // Inplace
    cpu_src1.Exp();
    gpu_src1.Exp();

    gpu_res << gpu_src1;

    CHECK(cpu_src1.IsSimilar(gpu_res, 0.0001f));
}
