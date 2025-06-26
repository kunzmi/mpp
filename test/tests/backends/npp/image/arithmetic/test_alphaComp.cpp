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
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32u.h>
#include <backends/npp/image/image32uC1View.h>
#include <backends/npp/image/image32uC2View.h>
#include <backends/npp/image/image32uC3View.h>
#include <backends/npp/image/image32uC4View.h>
#include <backends/npp/image/image8s.h>
#include <backends/npp/image/image8sC1View.h>
#include <backends/npp/image/image8sC2View.h>
#include <backends/npp/image/image8sC3View.h>
#include <backends/npp/image/image8sC4View.h>
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

TEST_CASE("8uC1", "[NPP.Arithmetic.AlphaCompC]")
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
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 alpha1;
    Pixel8uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("8uC3", "[NPP.Arithmetic.AlphaCompC]")
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
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 alpha1;
    Pixel8uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("8uC4", "[NPP.Arithmetic.AlphaCompC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::Image<Pixel8uC4> cpu_src2(size, size);
    cpu::Image<Pixel8uC4> cpu_dst(size, size);
    cpu::Image<Pixel8uC4> npp_res(size, size);
    nv::Image8uC4 npp_src1(size, size);
    nv::Image8uC4 npp_src2(size, size);
    nv::Image8uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 alpha1;
    Pixel8uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.AlphaCompC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

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

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    FillRandom<Pixel8uC1> op(seed + 1);
    Pixel8uC1 alpha1;
    Pixel8uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

// 8s is behaving weirdly, not sure if it is mpp or npp, but the results don't match
// TEST_CASE("8sC1", "[NPP.Arithmetic.AlphaCompC]")
//{
//     const uint seed         = Catch::getSeed();
//     NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();
//
//     cpu::Image<Pixel8sC1> cpu_src1(size, size);
//     cpu::Image<Pixel8sC1> cpu_src2(size, size);
//     cpu::Image<Pixel8sC1> cpu_dst(size, size);
//     cpu::Image<Pixel8sC1> npp_res(size, size);
//     nv::Image8sC1 npp_src1(size, size);
//     nv::Image8sC1 npp_src2(size, size);
//     nv::Image8sC1 npp_dst(size, size);
//
//     cpu_src1.FillRandom(seed);
//     cpu_src2.FillRandom(seed + 1);
//     FillRandom<Pixel8sC1> op(seed + 1);
//     Pixel8sC1 alpha1;
//     Pixel8sC1 alpha2;
//     op(alpha1);
//     op(alpha2);
//     alpha1.Abs();
//     alpha2.Abs();
//
//     cpu_src1 >> npp_src1;
//     cpu_src2 >> npp_src2;
//
//     // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
//
//     /*cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 3));*/
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
//
//     /*cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 3));*/
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
//
//     cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
//     npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);
//
//     npp_res << npp_dst;
//
//     CHECK(cpu_dst.IsSimilar(npp_res, 2));
// }

TEST_CASE("16uC1", "[NPP.Arithmetic.AlphaCompC]")
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
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 alpha1;
    Pixel16uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("16uC3", "[NPP.Arithmetic.AlphaCompC]")
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
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 alpha1;
    Pixel16uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("16uC4", "[NPP.Arithmetic.AlphaCompC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::Image<Pixel16uC4> cpu_src2(size, size);
    cpu::Image<Pixel16uC4> cpu_dst(size, size);
    cpu::Image<Pixel16uC4> npp_res(size, size);
    nv::Image16uC4 npp_src1(size, size);
    nv::Image16uC4 npp_src2(size, size);
    nv::Image16uC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 alpha1;
    Pixel16uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("16uC4A", "[NPP.Arithmetic.AlphaCompC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

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

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    FillRandom<Pixel16uC1> op(seed + 1);
    Pixel16uC1 alpha1;
    Pixel16uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1A.AlphaComp(cpu_src2A, cpu_dstA, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaCompA(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("16sC1", "[NPP.Arithmetic.AlphaCompC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_src2(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC1 npp_src2(size, size);
    nv::Image16sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    FillRandom<Pixel16sC1> op(seed + 1);
    Pixel16sC1 alpha1;
    Pixel16sC1 alpha2;
    op(alpha1);
    op(alpha2);
    alpha1.Abs(); // make sure only positive alpha values
    alpha2.Abs();

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...
    // but here NPP seems off by more in some cases

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 5));
}

TEST_CASE("32uC1", "[NPP.Arithmetic.AlphaCompC]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32uC1::GetStreamContext();

    cpu::Image<Pixel32uC1> cpu_src1(size, size);
    cpu::Image<Pixel32uC1> cpu_src2(size, size);
    cpu::Image<Pixel32uC1> cpu_dst(size, size);
    cpu::Image<Pixel32uC1> npp_res(size, size);
    nv::Image32uC1 npp_src1(size, size);
    nv::Image32uC1 npp_src2(size, size);
    nv::Image32uC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    cpu_src2.FillRandom(seed + 1);
    FillRandom<Pixel32uC1> op(seed + 1);
    Pixel32uC1 alpha1;
    Pixel32uC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 2));
}

TEST_CASE("32sC1", "[NPP.Arithmetic.AlphaCompC]")
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
    FillRandom<Pixel32sC1> op(seed + 1);
    Pixel32sC1 alpha1;
    Pixel32sC1 alpha2;
    op(alpha1);
    op(alpha2);
    alpha1.Abs();
    alpha2.Abs();

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 3));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.AlphaCompC]")
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
    FillRandom<Pixel32fC1> op(seed + 1);
    Pixel32fC1 alpha1;
    Pixel32fC1 alpha2;
    op(alpha1);
    op(alpha2);

    cpu_src1 >> npp_src1;
    cpu_src2 >> npp_src2;

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, alpha1.x, alpha2.x, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaComp(alpha1.x, npp_src2, alpha2.x, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}

TEST_CASE("8uC4A", "[NPP.Arithmetic.AlphaComp]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

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

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;
    // again, npp is buggy, instead of setting alpha to 255, it sometimes sets the value to 0
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;
    {
        auto iter_cpu = cpu_dst.begin();
        for (auto &npp_pixel : npp_res)
        {
            if (npp_pixel.Value().w == 0 && iter_cpu.Value().w == 255)
            {
                npp_pixel.Value().w = 255;
            }
            ++iter_cpu;
        }
    }

    CHECK(cpu_dst.IsSimilar(npp_res, 3));
}

TEST_CASE("32fC4A", "[NPP.Arithmetic.AlphaComp]")
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

    // due to rounding errors, each multiplication can be off by one, so the sum can be off by two in worst case...

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::ATop);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::ATopPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::In);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::InPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::Out);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::OutPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::Over);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::OverPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::Plus);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));

    cpu_src1.AlphaComp(cpu_src2, cpu_dst, mpp::AlphaCompositionOp::PlusPremul);
    npp_src1.AlphaCompA(npp_src2, npp_dst, NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsSimilar(npp_res, 0.00001f));
}