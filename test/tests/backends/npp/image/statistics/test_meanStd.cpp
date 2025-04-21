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

TEST_CASE("8uC1", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 15);

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.01));
}

TEST_CASE("8uC3", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 15);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.01));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.01));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.01));
}

TEST_CASE("8uC1", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 15);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.01));
}

TEST_CASE("8uC3", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 15);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, 1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.01));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.01));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.01));
}

TEST_CASE("8sC1", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image8sC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 15);

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.01));
}

TEST_CASE("8sC3", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image8sC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 15);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.01));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.01));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.01));
}

TEST_CASE("8sC1", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8sC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 0, 15);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.01));
}

TEST_CASE("8sC3", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image8sC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 127, 15);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, 1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.01));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.01));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.01));
}

TEST_CASE("16uC1", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image16uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.01));
}

TEST_CASE("16uC3", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image16uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.01));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.01));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.01));
}

TEST_CASE("16uC1", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image16uC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.01));
}

TEST_CASE("16uC3", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image16uC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, 1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.01));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.01));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.01));
}

TEST_CASE("32fC1", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.1));
}

TEST_CASE("32fC3", "[NPP.Statistics.MeanStd]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSize(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;

    npp_src1.Mean_StdDev(1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStd(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.1));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.1));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.1));
}

TEST_CASE("32fC1", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel64fC1 cpu_mean;
    Pixel64fC1 cpu_std;
    double npp_mean;
    double npp_std;
    nv::Image8uC1 npp_mask(size, size);
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(1);
    opp::cuda::DevVar<double> npp_dst2(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, npp_buffer, npp_dst1, npp_dst2, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std).margin(0.1));
}

TEST_CASE("32fC3", "[NPP.Statistics.MeanStdMasked]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    std::filesystem::path root     = std::filesystem::path(TEST_DATA_DIR);
    cpu::Image<Pixel8uC1> cpu_mask = cpu::Image<Pixel8uC1>::Load(root / "mask_random_0.5.tif");
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel64fC3 cpu_mean;
    Pixel64fC3 cpu_std;
    double cpu_meanScalar;
    double cpu_stdScalar;
    double npp_mean[3];
    double npp_std[3];
    nv::Image8uC1 npp_mask(size, size);
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<double> npp_dst1(3);
    opp::cuda::DevVarView<double> npp_dst11(npp_dst1.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst12(npp_dst1.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst13(npp_dst1.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<double> npp_dst2(3);
    opp::cuda::DevVarView<double> npp_dst21(npp_dst2.Pointer() + 0, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst22(npp_dst2.Pointer() + 1, sizeof(double));
    opp::cuda::DevVarView<double> npp_dst23(npp_dst2.Pointer() + 2, sizeof(double));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MeanStdDevGetBufferHostSizeMasked(nppCtx));

    cpu_src1.FillRandomNormal(seed, 30000, 150);
    cpu_src1.Mul({1, 2, 3});

    cpu_src1 >> npp_src1;
    cpu_mask >> npp_mask;

    npp_src1.Mean_StdDev(npp_mask, 1, npp_buffer, npp_dst11, npp_dst21, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 2, npp_buffer, npp_dst12, npp_dst22, nppCtx);
    npp_src1.Mean_StdDev(npp_mask, 3, npp_buffer, npp_dst13, npp_dst23, nppCtx);
    npp_dst1 >> npp_mean;
    npp_dst2 >> npp_std;

    cpu_src1.MeanStdMasked(cpu_mean, cpu_std, cpu_meanScalar, cpu_stdScalar, cpu_mask);

    CHECK(cpu_mean.x == Approx(npp_mean[0]).margin(0.01));
    CHECK(cpu_std.x == Approx(npp_std[0]).margin(0.1));
    CHECK(cpu_mean.y == Approx(npp_mean[1]).margin(0.01));
    CHECK(cpu_std.y == Approx(npp_std[1]).margin(0.1));
    CHECK(cpu_mean.z == Approx(npp_mean[2]).margin(0.01));
    CHECK(cpu_std.z == Approx(npp_std[2]).margin(0.1));
}