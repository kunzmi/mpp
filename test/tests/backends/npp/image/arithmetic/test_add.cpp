#include <backends/npp/image/image16f.h>
#include <backends/npp/image/image16fC1View.h>
#include <backends/npp/image/image16fC2View.h>
#include <backends/npp/image/image16fC3View.h>
#include <backends/npp/image/image16fC4View.h>
#include <backends/npp/image/image16sc.h>
#include <backends/npp/image/image16scC1View.h>
#include <backends/npp/image/image16scC2View.h>
#include <backends/npp/image/image16scC3View.h>
#include <backends/npp/image/image16scC4View.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC2View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image32fcC1View.h>
#include <backends/npp/image/image32fcC2View.h>
#include <backends/npp/image/image32fcC3View.h>
#include <backends/npp/image/image32fcC4View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sc.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32scC1View.h>
#include <backends/npp/image/image32scC2View.h>
#include <backends/npp/image/image32scC3View.h>
#include <backends/npp/image/image32scC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC2View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_src2(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_src2(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_src2(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_src2(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    nv::Image8uC4 npp_dst(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel8uC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel8uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Add(cpu_src2A, cpu_dstA);
    npp_src1.AddA(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Add(cpu_src2A, cpu_dstA, 2);
    npp_src1.AddA(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Add(cpu_src2A, cpu_dstA, 1);
    npp_dst.AddA(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_src2(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_src2(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_src2(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_src2(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    nv::Image16uC4 npp_dst(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16uC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel16uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Add(cpu_src2A, cpu_dstA);
    npp_src1.AddA(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Add(cpu_src2A, cpu_dstA, 2);
    npp_src1.AddA(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Add(cpu_src2A, cpu_dstA, 1);
    npp_dst.AddA(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC1", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_src2(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> npp_res(size, size);
    nv::Image32sC1 npp_src1(size, size);
    nv::Image32sC1 npp_src2(size, size);
    nv::Image32sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC3", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC3::GetStreamContext();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_src2(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> npp_res(size, size);
    nv::Image32sC3 npp_src1(size, size);
    nv::Image32sC3 npp_src2(size, size);
    nv::Image32sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_src2(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_src2(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2);
    npp_dst.Add(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC3", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC3::GetStreamContext();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_src2(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> npp_res(size, size);
    nv::Image16fC3 npp_src1(size, size);
    nv::Image16fC3 npp_src2(size, size);
    nv::Image16fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2);
    npp_dst.Add(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC4", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC4::GetStreamContext();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_src2(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> npp_res(size, size);
    nv::Image16fC4 npp_src1(size, size);
    nv::Image16fC4 npp_src2(size, size);
    nv::Image16fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2);
    npp_dst.Add(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_src2(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_src2(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2);
    npp_dst.Add(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_src2(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_src2(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2);
    npp_dst.Add(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_src2(size, size);
    nv::Image32fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2);
    npp_dst.Add(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_src2(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_src2(size, size);
    nv::Image32fC4 npp_dst(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel32fC4A> cpu_src2A = cpu_src2;
    cpu::ImageView<Pixel32fC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1A.Add(cpu_src2A, cpu_dstA);
    npp_src1.AddA(npp_src2, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Add(cpu_src2A, cpu_dstA);
    npp_dst.AddA(npp_src2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC1", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_src2(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> npp_res(size, size);
    nv::Image16scC1 npp_src1(size, size);
    nv::Image16scC1 npp_src2(size, size);
    nv::Image16scC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC3", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC3::GetStreamContext();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_src2(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> npp_res(size, size);
    nv::Image16scC3 npp_src1(size, size);
    nv::Image16scC3 npp_src2(size, size);
    nv::Image16scC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.Add(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.Add(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.Add(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC4", "[NPP.Arithmetic.Add]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC4::GetStreamContext();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_src2(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> npp_res(size, size);
    nv::Image16scC4 npp_src1(size, size);
    nv::Image16scC4 npp_src2(size, size);
    nv::Image16scC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.Add(cpu_src2, cpu_dst);
    npp_src1.AddA(npp_src2, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(cpu_src2, cpu_dst, 2);
    npp_src1.AddA(npp_src2, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(cpu_src2, 1);
    npp_dst.AddA(npp_src2, 1, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC1", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC3> op(seed + 1);
    Pixel8uC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel8uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel8uC4> op(seed + 1);
    Pixel8uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.Add(constVal, cpu_dstA);
    npp_src1.AddA(constVal.XYZ(), npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Add(constVal, cpu_dstA, 2);
    npp_src1.AddA(constVal.XYZ(), npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Add(constVal, cpu_dstA, 1);
    npp_dst.AddA(constVal.XYZ(), npp_dst, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC1", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC3> op(seed + 1);
    Pixel16uC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16uC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16uC4> op(seed + 1);
    Pixel16uC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.Add(constVal, cpu_dstA);
    npp_src1.AddA(constVal.XYZ(), npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Add(constVal, cpu_dstA, 2);
    npp_src1.AddA(constVal.XYZ(), npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Add(constVal, cpu_dstA, 1);
    npp_dst.AddA(constVal.XYZ(), npp_dst, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC1", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC1::GetStreamContext();

    cpu::Image<Pixel32sC1> cpu_src1(size, size);
    cpu::Image<Pixel32sC1> cpu_dst(size, size);
    cpu::Image<Pixel32sC1> npp_res(size, size);
    nv::Image32sC1 npp_src1(size, size);
    nv::Image32sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32sC3", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32sC3::GetStreamContext();

    cpu::Image<Pixel32sC3> cpu_src1(size, size);
    cpu::Image<Pixel32sC3> cpu_dst(size, size);
    cpu::Image<Pixel32sC3> npp_res(size, size);
    nv::Image32sC3 npp_src1(size, size);
    nv::Image32sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32sC3> op(seed + 1);
    Pixel32sC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC1> op(seed + 1);
    Pixel16fC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal);
    npp_dst.Add(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC3", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC3::GetStreamContext();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> npp_res(size, size);
    nv::Image16fC3 npp_src1(size, size);
    nv::Image16fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC3> op(seed + 1);
    Pixel16fC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal);
    npp_dst.Add(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16fC4", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC4::GetStreamContext();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> npp_res(size, size);
    nv::Image16fC4 npp_src1(size, size);
    nv::Image16fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16fC4> op(seed + 1);
    Pixel16fC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal);
    npp_dst.Add(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal);
    npp_dst.Add(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC3> op(seed + 1);
    Pixel32fC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal);
    npp_dst.Add(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal);
    npp_dst.Add(constVal, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel32fC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel32fC4> op(seed + 1);
    Pixel32fC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1A.Add(constVal, cpu_dstA);
    npp_src1.AddA(constVal.XYZ(), npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dstA.Add(constVal, cpu_dstA);
    npp_dst.AddA(constVal.XYZ(), npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC1", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC1::GetStreamContext();

    cpu::Image<Pixel16scC1> cpu_src1(size, size);
    cpu::Image<Pixel16scC1> cpu_dst(size, size);
    cpu::Image<Pixel16scC1> npp_res(size, size);
    nv::Image16scC1 npp_src1(size, size);
    nv::Image16scC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC1> op(seed + 1);
    Pixel16scC1 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC3", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC3::GetStreamContext();

    cpu::Image<Pixel16scC3> cpu_src1(size, size);
    cpu::Image<Pixel16scC3> cpu_dst(size, size);
    cpu::Image<Pixel16scC3> npp_res(size, size);
    nv::Image16scC3 npp_src1(size, size);
    nv::Image16scC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC3> op(seed + 1);
    Pixel16scC3 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.Add(constVal, npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.Add(constVal, npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.Add(constVal, 1, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));
}

TEST_CASE("16scC4", "[NPP.Arithmetic.AddC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16scC4::GetStreamContext();

    cpu::Image<Pixel16scC4> cpu_src1(size, size);
    cpu::Image<Pixel16scC4> cpu_dst(size, size);
    cpu::Image<Pixel16scC4> npp_res(size, size);
    nv::Image16scC4 npp_src1(size, size);
    nv::Image16scC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    FillRandom<Pixel16scC4> op(seed + 1);
    Pixel16scC4 constVal;
    op(constVal);

    cpu_src1 >> npp_src1;

    cpu_src1.Add(constVal, cpu_dst);
    npp_src1.AddA(constVal.XYZ(), npp_dst, 0, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Add(constVal, cpu_dst, 2);
    npp_src1.AddA(constVal.XYZ(), npp_dst, 2, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_dst.Add(constVal, 1);
    npp_dst.AddA(constVal.XYZ(), npp_dst, 1, nppCtx);

    npp_res << npp_dst;

    for (auto &elem : npp_res)
    {
        elem.Value().w = 0;
    }
    for (auto &elem : cpu_dst)
    {
        elem.Value().w = 0;
    }

    CHECK(cpu_dst.IsIdentical(npp_res));
}