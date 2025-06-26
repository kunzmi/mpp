#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC2View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC2View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
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
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    cpu::Image<Pixel8uC1> cpu_dst(size, size);
    cpu::Image<Pixel8uC1> npp_res(size, size);
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC3", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    cpu::Image<Pixel8uC3> cpu_dst(size, size);
    cpu::Image<Pixel8uC3> npp_res(size, size);
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC3 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("8uC4", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC1", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    cpu::Image<Pixel16uC1> cpu_dst(size, size);
    cpu::Image<Pixel16uC1> npp_res(size, size);
    nv::Image16uC1 npp_src1(size, size);
    nv::Image16uC1 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC3", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    cpu::Image<Pixel16uC3> cpu_dst(size, size);
    cpu::Image<Pixel16uC3> npp_res(size, size);
    nv::Image16uC3 npp_src1(size, size);
    nv::Image16uC3 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16uC4", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16sC1", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC1 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16sC3", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> npp_res(size, size);
    nv::Image16sC3 npp_src1(size, size);
    nv::Image16sC3 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("16sC4", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder32f(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    // NPP doesn't round when converting float to int
    CHECK(cpu_dst.IsSimilar(npp_res, 1));
}

TEST_CASE("32fC1", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.0001f));
}

TEST_CASE("32fC3", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.0001f));
}

TEST_CASE("32fC4", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.0001f));
}

TEST_CASE("32fC4A", "[NPP.Filtering.RowFilter]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4A> cpu_src1A(size, size);
    cpu::ImageView<Pixel32fC4> cpu_src1(cpu_src1A);
    cpu::Image<Pixel32fC4A> cpu_dst(size, size);
    nv::Image32fC4 npp_src1(size, size);

    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_dst(size, size);
    cpu::Image<Pixel32fC1> filter(5, 1);
    mpp::cuda::DevVar<float> filterd(5 * 1);

    cpu_src1.FillRandom(seed);
    filter.FillRandom(seed + 1);
    filter.Sub(0.5f);
    filterd << filter.Pointer();
    filter.Mirror(MirrorAxis::Both);

    cpu_src1 >> npp_src1;

    npp_src1.FilterRowBorder(npp_dst, filterd, 5, 2, NPP_BORDER_REPLICATE, nppCtx);
    npp_res << npp_dst;

    cpu_src1A.RowFilter(cpu_dst, reinterpret_cast<float *>(filter.Pointer()), 5, 2, BorderType::Replicate);

    CHECK(cpu_dst.IsSimilar(npp_res, 0.0001f));
}
