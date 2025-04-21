#include <backends/cuda/devVar.h>
#include <backends/cuda/devVarView.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image8s.h>
#include <backends/npp/image/image8sC1View.h>
#include <backends/npp/image/image8sC3View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/statistics/indexMinMax.h>
#include <utility>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 cpu_dstMax;
    byte npp_resMin;
    byte npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image8uC1 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(1);
    opp::cuda::DevVar<byte> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 11)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == 4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 200);
    CHECK(cpu_idx.IndexMin.x == 100);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("8uC3", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel8uC3 cpu_dstMin;
    Pixel8uC3 cpu_dstMax;
    byte cpu_dstScalarMin;
    byte cpu_dstScalarMax;
    byte npp_resMin[3];
    byte npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image8uC3 npp_src1(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(3);
    opp::cuda::DevVar<byte> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<byte> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(byte));
    opp::cuda::DevVarView<byte> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(byte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<byte> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(byte));
    opp::cuda::DevVarView<byte> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(byte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<byte> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(byte));
    opp::cuda::DevVarView<byte> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(byte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == 4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 200);
    CHECK(cpu_idx[0].IndexMin.x == 100);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == 5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 201);
    CHECK(cpu_idx[1].IndexMin.x == 100);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == 6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 202);
    CHECK(cpu_idx[2].IndexMin.x == 100);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("8sC1", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    Pixel8sC1 cpu_dstMin;
    Pixel8sC1 cpu_dstMax;
    sbyte npp_resMin;
    sbyte npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image8sC1 npp_src1(size, size);
    opp::cuda::DevVar<sbyte> npp_dstMin(1);
    opp::cuda::DevVar<sbyte> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(0);
    cpu_src1(10, 11)  = 127;
    cpu_src1(100, 10) = -4;
    cpu_src1(104, 10) = -4;
    cpu_src1(10, 100) = -4;
    cpu_src1(9, 120)  = -4;

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == -4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 127);
    CHECK(cpu_idx.IndexMin.x == 100);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("8sC3", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image8sC3::GetStreamContext();

    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    Pixel8sC3 cpu_dstMin;
    Pixel8sC3 cpu_dstMax;
    sbyte cpu_dstScalarMin;
    sbyte cpu_dstScalarMax;
    sbyte npp_resMin[3];
    sbyte npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image8sC3 npp_src1(size, size);
    opp::cuda::DevVar<sbyte> npp_dstMin(3);
    opp::cuda::DevVar<sbyte> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<sbyte> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(sbyte));
    opp::cuda::DevVarView<sbyte> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(sbyte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<sbyte> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(sbyte));
    opp::cuda::DevVarView<sbyte> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(sbyte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<sbyte> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(sbyte));
    opp::cuda::DevVarView<sbyte> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(sbyte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(0);
    cpu_src1(10, 10)  = {120, 121, 122};
    cpu_src1(100, 10) = {-4, -5, -6};
    cpu_src1(104, 10) = {-4, -5, -6};
    cpu_src1(10, 100) = {-4, -5, -6};
    cpu_src1(9, 120)  = {-4, -5, -6};

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == -4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 120);
    CHECK(cpu_idx[0].IndexMin.x == 100);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == -5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 121);
    CHECK(cpu_idx[1].IndexMin.x == 100);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == -6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 122);
    CHECK(cpu_idx[2].IndexMin.x == 100);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("16uC1", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 cpu_dstMax;
    ushort npp_resMin;
    ushort npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image16uC1 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(1);
    opp::cuda::DevVar<ushort> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 11)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == 4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 200);
    CHECK(cpu_idx.IndexMin.x == 100);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("16uC3", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel16uC3 cpu_dstMin;
    Pixel16uC3 cpu_dstMax;
    ushort cpu_dstScalarMin;
    ushort cpu_dstScalarMax;
    ushort npp_resMin[3];
    ushort npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image16uC3 npp_src1(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(3);
    opp::cuda::DevVar<ushort> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<ushort> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(ushort));
    opp::cuda::DevVarView<ushort> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(ushort));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<ushort> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(ushort));
    opp::cuda::DevVarView<ushort> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(ushort));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<ushort> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(ushort));
    opp::cuda::DevVarView<ushort> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(ushort));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == 4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 200);
    CHECK(cpu_idx[0].IndexMin.x == 100);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == 5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 201);
    CHECK(cpu_idx[1].IndexMin.x == 100);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == 6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 202);
    CHECK(cpu_idx[2].IndexMin.x == 100);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("32fC1", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 cpu_dstMax;
    float npp_resMin;
    float npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image32fC1 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dstMin(1);
    opp::cuda::DevVar<float> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 11)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == 4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 200);
    CHECK(cpu_idx.IndexMin.x == 100);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("32fC3", "[NPP.Statistics.MinMaxIndex]")
{
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel32fC3 cpu_dstMin;
    Pixel32fC3 cpu_dstMax;
    float cpu_dstScalarMin;
    float cpu_dstScalarMax;
    float npp_resMin[3];
    float npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image32fC3 npp_src1(size, size);
    opp::cuda::DevVar<float> npp_dstMin(3);
    opp::cuda::DevVar<float> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<float> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(float));
    opp::cuda::DevVarView<float> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(float));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<float> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(float));
    opp::cuda::DevVarView<float> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(float));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<float> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(float));
    opp::cuda::DevVarView<float> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(float));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndex(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == 4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 200);
    CHECK(cpu_idx[0].IndexMin.x == 100);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == 5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 201);
    CHECK(cpu_idx[1].IndexMin.x == 100);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == 6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 202);
    CHECK(cpu_idx[2].IndexMin.x == 100);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("8uC1", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel8uC1 cpu_dstMin;
    Pixel8uC1 cpu_dstMax;
    byte npp_resMin;
    byte npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image8uC1 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(1);
    opp::cuda::DevVar<byte> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 255;
    cpu_mask(10, 100) = 255;
    cpu_mask(9, 120)  = 255;
    cpu_src1.Set(127);
    cpu_src1(9, 10)   = 200;
    cpu_src1(10, 11)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_mask);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == 4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 200);
    CHECK(cpu_idx.IndexMin.x == 104);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("8uC3", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel8uC3 cpu_dstMin;
    Pixel8uC3 cpu_dstMax;
    byte cpu_dstScalarMin;
    byte cpu_dstScalarMax;
    byte npp_resMin[3];
    byte npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image8uC3 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<byte> npp_dstMin(3);
    opp::cuda::DevVar<byte> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<byte> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(byte));
    opp::cuda::DevVarView<byte> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(byte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<byte> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(byte));
    opp::cuda::DevVarView<byte> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(byte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<byte> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(byte));
    opp::cuda::DevVarView<byte> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(byte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 4;
    cpu_mask(10, 100) = 4;
    cpu_mask(9, 120)  = 4;
    cpu_src1.Set(127);
    cpu_src1(9, 10)   = {200, 201, 202};
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, 1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar,
                               cpu_mask);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == 4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 200);
    CHECK(cpu_idx[0].IndexMin.x == 104);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == 5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 201);
    CHECK(cpu_idx[1].IndexMin.x == 104);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == 6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 202);
    CHECK(cpu_idx[2].IndexMin.x == 104);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("8sC1", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image8sC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel8sC1> cpu_src1(size, size);
    Pixel8sC1 cpu_dstMin;
    Pixel8sC1 cpu_dstMax;
    sbyte npp_resMin;
    sbyte npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image8sC1 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<sbyte> npp_dstMin(1);
    opp::cuda::DevVar<sbyte> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 255;
    cpu_mask(10, 100) = 255;
    cpu_mask(9, 120)  = 255;
    cpu_src1.Set(0);
    cpu_src1(9, 10)   = 127;
    cpu_src1(10, 11)  = 127;
    cpu_src1(100, 10) = -4;
    cpu_src1(104, 10) = -4;
    cpu_src1(10, 100) = -4;
    cpu_src1(9, 120)  = -4;

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_mask);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == -4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 127);
    CHECK(cpu_idx.IndexMin.x == 104);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("8sC3", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image8sC3::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel8sC3> cpu_src1(size, size);
    Pixel8sC3 cpu_dstMin;
    Pixel8sC3 cpu_dstMax;
    sbyte cpu_dstScalarMin;
    sbyte cpu_dstScalarMax;
    sbyte npp_resMin[3];
    sbyte npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image8sC3 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<sbyte> npp_dstMin(3);
    opp::cuda::DevVar<sbyte> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<sbyte> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(sbyte));
    opp::cuda::DevVarView<sbyte> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(sbyte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<sbyte> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(sbyte));
    opp::cuda::DevVarView<sbyte> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(sbyte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<sbyte> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(sbyte));
    opp::cuda::DevVarView<sbyte> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(sbyte));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 255;
    cpu_mask(10, 100) = 255;
    cpu_mask(9, 120)  = 255;
    cpu_src1.Set(0);
    cpu_src1(9, 10)   = {120, 121, 122};
    cpu_src1(10, 10)  = {120, 121, 122};
    cpu_src1(100, 10) = {-4, -5, -6};
    cpu_src1(104, 10) = {-4, -5, -6};
    cpu_src1(10, 100) = {-4, -5, -6};
    cpu_src1(9, 120)  = {-4, -5, -6};

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, 1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar,
                               cpu_mask);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == -4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 120);
    CHECK(cpu_idx[0].IndexMin.x == 104);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == -5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 121);
    CHECK(cpu_idx[1].IndexMin.x == 104);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == -6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 122);
    CHECK(cpu_idx[2].IndexMin.x == 104);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("16uC1", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel16uC1 cpu_dstMin;
    Pixel16uC1 cpu_dstMax;
    ushort npp_resMin;
    ushort npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image16uC1 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(1);
    opp::cuda::DevVar<ushort> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 255;
    cpu_mask(10, 100) = 255;
    cpu_mask(9, 120)  = 255;
    cpu_src1.Set(127);
    cpu_src1(9, 10)   = 200;
    cpu_src1(10, 11)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_mask);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == 4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 200);
    CHECK(cpu_idx.IndexMin.x == 104);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("16uC3", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel16uC3 cpu_dstMin;
    Pixel16uC3 cpu_dstMax;
    ushort cpu_dstScalarMin;
    ushort cpu_dstScalarMax;
    ushort npp_resMin[3];
    ushort npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image16uC3 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<ushort> npp_dstMin(3);
    opp::cuda::DevVar<ushort> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<ushort> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(ushort));
    opp::cuda::DevVarView<ushort> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(ushort));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<ushort> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(ushort));
    opp::cuda::DevVarView<ushort> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(ushort));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<ushort> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(ushort));
    opp::cuda::DevVarView<ushort> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(ushort));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 4;
    cpu_mask(10, 100) = 4;
    cpu_mask(9, 120)  = 4;
    cpu_src1.Set(127);
    cpu_src1(9, 10)   = {200, 201, 202};
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, 1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar,
                               cpu_mask);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == 4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 200);
    CHECK(cpu_idx[0].IndexMin.x == 104);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == 5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 201);
    CHECK(cpu_idx[1].IndexMin.x == 104);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == 6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 202);
    CHECK(cpu_idx[2].IndexMin.x == 104);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}

TEST_CASE("32fC1", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel32fC1 cpu_dstMin;
    Pixel32fC1 cpu_dstMax;
    float npp_resMin;
    float npp_resMax;
    IndexMinMax cpu_idx;
    NppiPoint npp_idxMin;
    NppiPoint npp_idxMax;
    nv::Image32fC1 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<float> npp_dstMin(1);
    opp::cuda::DevVar<float> npp_dstMax(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(1);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(1);
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 255;
    cpu_mask(10, 100) = 255;
    cpu_mask(9, 120)  = 255;
    cpu_src1.Set(127);
    cpu_src1(9, 10)   = 200;
    cpu_src1(10, 11)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, npp_dstMin, npp_dstMax, npp_dstIdxMin, npp_dstIdxMax, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_mask);

    CHECK(cpu_src1(cpu_idx.IndexMin.x, cpu_idx.IndexMin.y) == 4);
    CHECK(cpu_src1(cpu_idx.IndexMax.x, cpu_idx.IndexMax.y) == 200);
    CHECK(cpu_idx.IndexMin.x == 104);
    CHECK(cpu_idx.IndexMin.y == 10);
    CHECK(cpu_idx.IndexMax.x == 10);
    CHECK(cpu_idx.IndexMax.y == 11);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx.IndexMin.x == npp_idxMin.x);
    CHECK(cpu_idx.IndexMin.y == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 200);*/
    CHECK(cpu_dstMin.x == npp_resMin);
    CHECK(cpu_dstMax.x == npp_resMax);
    CHECK(npp_idxMax.x == 10);
    CHECK(npp_idxMax.y == 11);
}

TEST_CASE("32fC3", "[NPP.Statistics.MinMaxIndexMasked]")
{
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_mask(size, size);
    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel32fC3 cpu_dstMin;
    Pixel32fC3 cpu_dstMax;
    float cpu_dstScalarMin;
    float cpu_dstScalarMax;
    float npp_resMin[3];
    float npp_resMax[3];
    IndexMinMaxChannel cpu_IdxScalar;
    IndexMinMax cpu_idx[3];
    NppiPoint npp_idxMin[3];
    NppiPoint npp_idxMax[3];
    nv::Image32fC3 npp_src1(size, size);
    nv::Image8uC1 npp_mask(size, size);
    opp::cuda::DevVar<float> npp_dstMin(3);
    opp::cuda::DevVar<float> npp_dstMax(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMin(3);
    opp::cuda::DevVar<NppiPoint> npp_dstIdxMax(3);
    opp::cuda::DevVarView<float> npp_dstMin1(npp_dstMin.Pointer() + 0, sizeof(float));
    opp::cuda::DevVarView<float> npp_dstMax1(npp_dstMax.Pointer() + 0, sizeof(float));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin1(npp_dstIdxMin.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax1(npp_dstIdxMax.Pointer() + 0, sizeof(NppiPoint));
    opp::cuda::DevVarView<float> npp_dstMin2(npp_dstMin.Pointer() + 1, sizeof(float));
    opp::cuda::DevVarView<float> npp_dstMax2(npp_dstMax.Pointer() + 1, sizeof(float));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin2(npp_dstIdxMin.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax2(npp_dstIdxMax.Pointer() + 1, sizeof(NppiPoint));
    opp::cuda::DevVarView<float> npp_dstMin3(npp_dstMin.Pointer() + 2, sizeof(float));
    opp::cuda::DevVarView<float> npp_dstMax3(npp_dstMax.Pointer() + 2, sizeof(float));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMin3(npp_dstIdxMin.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVarView<NppiPoint> npp_dstIdxMax3(npp_dstIdxMax.Pointer() + 2, sizeof(NppiPoint));
    opp::cuda::DevVar<byte> npp_buffer(npp_src1.MinMaxIndxGetBufferHostSize(nppCtx));

    cpu_mask.Set(255);
    cpu_mask(9, 10)   = 0;
    cpu_mask(100, 10) = 0;
    cpu_mask(104, 10) = 4;
    cpu_mask(10, 100) = 4;
    cpu_mask(9, 120)  = 4;
    cpu_src1.Set(127);
    cpu_src1(9, 10)   = {200, 201, 202};
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_mask >> npp_mask;
    cpu_src1 >> npp_src1;

    npp_src1.MinMaxIndx(npp_mask, 1, npp_dstMin1, npp_dstMax1, npp_dstIdxMin1, npp_dstIdxMax1, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 2, npp_dstMin2, npp_dstMax2, npp_dstIdxMin2, npp_dstIdxMax2, npp_buffer, nppCtx);
    npp_src1.MinMaxIndx(npp_mask, 3, npp_dstMin3, npp_dstMax3, npp_dstIdxMin3, npp_dstIdxMax3, npp_buffer, nppCtx);
    npp_dstMin >> npp_resMin;
    npp_dstMax >> npp_resMax;
    npp_dstIdxMin >> npp_idxMin;
    npp_dstIdxMax >> npp_idxMax;

    cpu_src1.MinMaxIndexMasked(cpu_dstMin, cpu_dstMax, cpu_idx, cpu_dstScalarMin, cpu_dstScalarMax, cpu_IdxScalar,
                               cpu_mask);

    CHECK(cpu_src1(cpu_idx[0].IndexMin.x, cpu_idx[0].IndexMin.y).x == 4);
    CHECK(cpu_src1(cpu_idx[0].IndexMax.x, cpu_idx[0].IndexMax.y).x == 200);
    CHECK(cpu_idx[0].IndexMin.x == 104);
    CHECK(cpu_idx[0].IndexMin.y == 10);
    CHECK(cpu_idx[0].IndexMax.x == 10);
    CHECK(cpu_idx[0].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[1].IndexMin.x, cpu_idx[1].IndexMin.y).y == 5);
    CHECK(cpu_src1(cpu_idx[1].IndexMax.x, cpu_idx[1].IndexMax.y).y == 201);
    CHECK(cpu_idx[1].IndexMin.x == 104);
    CHECK(cpu_idx[1].IndexMin.y == 10);
    CHECK(cpu_idx[1].IndexMax.x == 10);
    CHECK(cpu_idx[1].IndexMax.y == 10);

    CHECK(cpu_src1(cpu_idx[2].IndexMin.x, cpu_idx[2].IndexMin.y).z == 6);
    CHECK(cpu_src1(cpu_idx[2].IndexMax.x, cpu_idx[2].IndexMax.y).z == 202);
    CHECK(cpu_idx[2].IndexMin.x == 104);
    CHECK(cpu_idx[2].IndexMin.y == 10);
    CHECK(cpu_idx[2].IndexMax.x == 10);
    CHECK(cpu_idx[2].IndexMax.y == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idx[0].x == npp_idxMin.x);
    CHECK(cpu_idx[0].x == npp_idxMin.y);
    CHECK(cpu_src1(npp_idxMin.x, npp_idxMin.y) == 4);*/
    CHECK(cpu_dstScalarMin == std::min({npp_resMin[0], npp_resMin[1], npp_resMin[2]}));
    CHECK(cpu_dstScalarMax == std::max({npp_resMax[0], npp_resMax[1], npp_resMax[2]}));
    CHECK(cpu_dstMin.x == npp_resMin[0]);
    CHECK(cpu_dstMin.y == npp_resMin[1]);
    CHECK(cpu_dstMin.z == npp_resMin[2]);
    CHECK(cpu_dstMax.x == npp_resMax[0]);
    CHECK(cpu_dstMax.y == npp_resMax[1]);
    CHECK(cpu_dstMax.z == npp_resMax[2]);
}