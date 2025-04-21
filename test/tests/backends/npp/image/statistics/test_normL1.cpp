#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sc.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC2View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
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
#include <backends/npp/image/image32u.h>
#include <backends/npp/image/image32uC1View.h>
#include <backends/npp/image/image32uC2View.h>
#include <backends/npp/image/image32uC3View.h>
#include <backends/npp/image/image32uC4View.h>
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image64fC1View.h>
#include <backends/npp/image/image64fC2View.h>
#include <backends/npp/image/image64fC3View.h>
#include <backends/npp/image/image64fC4View.h>
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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8uC3", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}

TEST_CASE("8uC4", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res[4];
    nv::Image8uC4 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
    CHECK(cpu_dst.w == Approx(npp_res[3]).margin(0.00001));
}

TEST_CASE("8uC1", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8uC3", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVarView<double> npp_dst1(npp_dst.Pointer(), sizeof(double));
    opp::cuda::DevVarView<double> npp_dst2(npp_dst.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst3(npp_dst.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, 1, npp_dst1, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 2, npp_dst2, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 3, npp_dst3, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}

TEST_CASE("8sC1", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8sC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("8sC3", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8sC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVarView<double> npp_dst1(npp_dst.Pointer(), sizeof(double));
    opp::cuda::DevVarView<double> npp_dst2(npp_dst.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst3(npp_dst.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, 1, npp_dst1, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 2, npp_dst2, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 3, npp_dst3, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}

TEST_CASE("16uC1", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image16uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16uC3", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image16uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}

TEST_CASE("16uC4", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res[4];
    nv::Image16uC4 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
    CHECK(cpu_dst.w == Approx(npp_res[3]).margin(0.00001));
}

TEST_CASE("16uC1", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image16uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16uC3", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image16uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVarView<double> npp_dst1(npp_dst.Pointer(), sizeof(double));
    opp::cuda::DevVarView<double> npp_dst2(npp_dst.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst3(npp_dst.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, 1, npp_dst1, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 2, npp_dst2, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 3, npp_dst3, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}

TEST_CASE("16sC1", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image16sC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("16sC3", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image16sC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}

TEST_CASE("16sC4", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res[4];
    nv::Image16sC4 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
    CHECK(cpu_dst.w == Approx(npp_res[3]).margin(0.00001));
}

TEST_CASE("32fC1", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fC3", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}

TEST_CASE("32fC4", "[NPP.Statistics.NormL1]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    Pixel64fC4 cpu_dst;
    double cpu_dstScalar;
    double npp_res[4];
    nv::Image32fC4 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(4);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSize(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Norm_L1(npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1(cpu_dst, cpu_dstScalar);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
    CHECK(cpu_dst.w == Approx(npp_res[3]).margin(0.00001));
}

TEST_CASE("32fC1", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel64fC1 cpu_dst;
    double npp_res;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, npp_dst, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res).margin(0.00001));
}

TEST_CASE("32fC3", "[NPP.Statistics.NormL1Masked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel64fC3 cpu_dst;
    double cpu_dstScalar;
    double npp_res[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst(3);
    opp::cuda::DevVarView<double> npp_dst1(npp_dst.Pointer(), sizeof(double));
    opp::cuda::DevVarView<double> npp_dst2(npp_dst.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst3(npp_dst.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.NormL1GetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Norm_L1(npp_mask, 1, npp_dst1, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 2, npp_dst2, npp_buffer, nppCtx);
    npp_src1.Norm_L1(npp_mask, 3, npp_dst3, npp_buffer, nppCtx);
    npp_dst >> npp_res;

    cpu_src1.NormL1Masked(cpu_dst, cpu_dstScalar, cpu_mask);

    CHECK(cpu_dst.x == Approx(npp_res[0]).margin(0.00001));
    CHECK(cpu_dst.y == Approx(npp_res[1]).margin(0.00001));
    CHECK(cpu_dst.z == Approx(npp_res[2]).margin(0.00001));
}
