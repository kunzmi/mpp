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
#include <utility>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 256;

TEST_CASE("8uC1", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    Pixel8uC1 cpu_dst;
    byte npp_res;
    Pixel32sC1 cpu_idxX;
    Pixel32sC1 cpu_idxY;
    Pixel32sC1 npp_idxX;
    Pixel32sC1 npp_idxY;
    nv::Image8uC1 npp_src1(size, size);
    mpp::cuda::DevVar<byte> npp_dst(1);
    mpp::cuda::DevVar<int> npp_dstIdxX(1);
    mpp::cuda::DevVar<int> npp_dstIdxY(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x) == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 200);*/
    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("8uC3", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image8uC3::GetStreamContext();

    cpu::Image<Pixel8uC3> cpu_src1(size, size);
    Pixel8uC3 cpu_dst;
    byte cpu_dstScalar;
    byte npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC3 cpu_idxX;
    Pixel32sC3 cpu_idxY;
    Pixel32sC3 npp_idxX;
    Pixel32sC3 npp_idxY;
    nv::Image8uC3 npp_src1(size, size);
    mpp::cuda::DevVar<byte> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 200);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("8uC4", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    Pixel8uC4 cpu_dst;
    byte cpu_dstScalar;
    byte npp_res[4];
    Vec3i cpu_IdxScalar;
    Pixel32sC4 cpu_idxX;
    Pixel32sC4 cpu_idxY;
    Pixel32sC4 npp_idxX;
    Pixel32sC4 npp_idxY;
    nv::Image8uC4 npp_src1(size, size);
    mpp::cuda::DevVar<byte> npp_dst(4);
    mpp::cuda::DevVar<int> npp_dstIdxX(4);
    mpp::cuda::DevVar<int> npp_dstIdxY(4);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202, 203};
    cpu_src1(100, 10) = {4, 5, 6, 7};
    cpu_src1(104, 10) = {4, 5, 6, 7};
    cpu_src1(10, 100) = {4, 5, 6, 7};
    cpu_src1(9, 120)  = {4, 5, 6, 7};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);
    CHECK(cpu_src1(cpu_idxX.w, cpu_idxY.w).w == 7);
    CHECK(cpu_idxX.w == 100);
    CHECK(cpu_idxY.w == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2], npp_res[3]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("8uC4A", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image8uC4::GetStreamContext();

    cpu::Image<Pixel8uC4> cpu_src1(size, size);
    cpu::ImageView<Pixel8uC4A> cpu_src1A(cpu_src1);
    Pixel8uC4A cpu_dst;
    byte cpu_dstScalar;
    byte npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC4A cpu_idxX;
    Pixel32sC4A cpu_idxY;
    Pixel32sC4A npp_idxX;
    Pixel32sC4A npp_idxY;
    nv::Image8uC4 npp_src1(size, size);
    mpp::cuda::DevVar<byte> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSizeA(nppCtx));

    cpu_src1.Set(127);
    cpu_src1A(10, 10)  = {200, 201, 202};
    cpu_src1A(100, 10) = {4, 5, 6};
    cpu_src1A(104, 10) = {4, 5, 6};
    cpu_src1A(10, 100) = {4, 5, 6};
    cpu_src1A(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndxA(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1A.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("16sC1", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    Pixel16sC1 cpu_dst;
    short npp_res;
    Pixel32sC1 cpu_idxX;
    Pixel32sC1 cpu_idxY;
    Pixel32sC1 npp_idxX;
    Pixel32sC1 npp_idxY;
    nv::Image16sC1 npp_src1(size, size);
    mpp::cuda::DevVar<short> npp_dst(1);
    mpp::cuda::DevVar<int> npp_dstIdxX(1);
    mpp::cuda::DevVar<int> npp_dstIdxY(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x) == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("16sC3", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    Pixel16sC3 cpu_dst;
    short cpu_dstScalar;
    short npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC3 cpu_idxX;
    Pixel32sC3 cpu_idxY;
    Pixel32sC3 npp_idxX;
    Pixel32sC3 npp_idxY;
    nv::Image16sC3 npp_src1(size, size);
    mpp::cuda::DevVar<short> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 200);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("16sC4", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    Pixel16sC4 cpu_dst;
    short cpu_dstScalar;
    short npp_res[4];
    Vec3i cpu_IdxScalar;
    Pixel32sC4 cpu_idxX;
    Pixel32sC4 cpu_idxY;
    Pixel32sC4 npp_idxX;
    Pixel32sC4 npp_idxY;
    nv::Image16sC4 npp_src1(size, size);
    mpp::cuda::DevVar<short> npp_dst(4);
    mpp::cuda::DevVar<int> npp_dstIdxX(4);
    mpp::cuda::DevVar<int> npp_dstIdxY(4);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202, 203};
    cpu_src1(100, 10) = {4, 5, 6, 7};
    cpu_src1(104, 10) = {4, 5, 6, 7};
    cpu_src1(10, 100) = {4, 5, 6, 7};
    cpu_src1(9, 120)  = {4, 5, 6, 7};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);
    CHECK(cpu_src1(cpu_idxX.w, cpu_idxY.w).w == 7);
    CHECK(cpu_idxX.w == 100);
    CHECK(cpu_idxY.w == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2], npp_res[3]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("16sC4A", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::ImageView<Pixel16sC4A> cpu_src1A(cpu_src1);
    Pixel16sC4A cpu_dst;
    short cpu_dstScalar;
    short npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC4A cpu_idxX;
    Pixel32sC4A cpu_idxY;
    Pixel32sC4A npp_idxX;
    Pixel32sC4A npp_idxY;
    nv::Image16sC4 npp_src1(size, size);
    mpp::cuda::DevVar<short> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSizeA(nppCtx));

    cpu_src1.Set(127);
    cpu_src1A(10, 10)  = {200, 201, 202};
    cpu_src1A(100, 10) = {4, 5, 6};
    cpu_src1A(104, 10) = {4, 5, 6};
    cpu_src1A(10, 100) = {4, 5, 6};
    cpu_src1A(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndxA(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1A.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("16uC1", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16uC1::GetStreamContext();

    cpu::Image<Pixel16uC1> cpu_src1(size, size);
    Pixel16uC1 cpu_dst;
    ushort npp_res;
    Pixel32sC1 cpu_idxX;
    Pixel32sC1 cpu_idxY;
    Pixel32sC1 npp_idxX;
    Pixel32sC1 npp_idxY;
    nv::Image16uC1 npp_src1(size, size);
    mpp::cuda::DevVar<ushort> npp_dst(1);
    mpp::cuda::DevVar<int> npp_dstIdxX(1);
    mpp::cuda::DevVar<int> npp_dstIdxY(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x) == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("16uC3", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16uC3::GetStreamContext();

    cpu::Image<Pixel16uC3> cpu_src1(size, size);
    Pixel16uC3 cpu_dst;
    ushort cpu_dstScalar;
    ushort npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC3 cpu_idxX;
    Pixel32sC3 cpu_idxY;
    Pixel32sC3 npp_idxX;
    Pixel32sC3 npp_idxY;
    nv::Image16uC3 npp_src1(size, size);
    mpp::cuda::DevVar<ushort> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("16uC4", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    Pixel16uC4 cpu_dst;
    ushort cpu_dstScalar;
    ushort npp_res[4];
    Vec3i cpu_IdxScalar;
    Pixel32sC4 cpu_idxX;
    Pixel32sC4 cpu_idxY;
    Pixel32sC4 npp_idxX;
    Pixel32sC4 npp_idxY;
    nv::Image16uC4 npp_src1(size, size);
    mpp::cuda::DevVar<ushort> npp_dst(4);
    mpp::cuda::DevVar<int> npp_dstIdxX(4);
    mpp::cuda::DevVar<int> npp_dstIdxY(4);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202, 203};
    cpu_src1(100, 10) = {4, 5, 6, 7};
    cpu_src1(104, 10) = {4, 5, 6, 7};
    cpu_src1(10, 100) = {4, 5, 6, 7};
    cpu_src1(9, 120)  = {4, 5, 6, 7};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);
    CHECK(cpu_src1(cpu_idxX.w, cpu_idxY.w).w == 7);
    CHECK(cpu_idxX.w == 100);
    CHECK(cpu_idxY.w == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2], npp_res[3]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("16uC4A", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image16uC4::GetStreamContext();

    cpu::Image<Pixel16uC4> cpu_src1(size, size);
    cpu::ImageView<Pixel16uC4A> cpu_src1A(cpu_src1);
    Pixel16uC4A cpu_dst;
    ushort cpu_dstScalar;
    ushort npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC4A cpu_idxX;
    Pixel32sC4A cpu_idxY;
    Pixel32sC4A npp_idxX;
    Pixel32sC4A npp_idxY;
    nv::Image16uC4 npp_src1(size, size);
    mpp::cuda::DevVar<ushort> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSizeA(nppCtx));

    cpu_src1.Set(127);
    cpu_src1A(10, 10)  = {200, 201, 202};
    cpu_src1A(100, 10) = {4, 5, 6};
    cpu_src1A(104, 10) = {4, 5, 6};
    cpu_src1A(10, 100) = {4, 5, 6};
    cpu_src1A(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndxA(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1A.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("32fC1", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    Pixel32fC1 cpu_dst;
    float npp_res;
    Pixel32sC1 cpu_idxX;
    Pixel32sC1 cpu_idxY;
    Pixel32sC1 npp_idxX;
    Pixel32sC1 npp_idxY;
    nv::Image32fC1 npp_src1(size, size);
    mpp::cuda::DevVar<float> npp_dst(1);
    mpp::cuda::DevVar<int> npp_dstIdxX(1);
    mpp::cuda::DevVar<int> npp_dstIdxY(1);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = 200;
    cpu_src1(100, 10) = 4;
    cpu_src1(104, 10) = 4;
    cpu_src1(10, 100) = 4;
    cpu_src1(9, 120)  = 4;

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x) == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dst.x == npp_res);
}

TEST_CASE("32fC3", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    Pixel32fC3 cpu_dst;
    float cpu_dstScalar;
    float npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC3 cpu_idxX;
    Pixel32sC3 cpu_idxY;
    Pixel32sC3 npp_idxX;
    Pixel32sC3 npp_idxY;
    nv::Image32fC3 npp_src1(size, size);
    mpp::cuda::DevVar<float> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202};
    cpu_src1(100, 10) = {4, 5, 6};
    cpu_src1(104, 10) = {4, 5, 6};
    cpu_src1(10, 100) = {4, 5, 6};
    cpu_src1(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}

TEST_CASE("32fC4", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    Pixel32fC4 cpu_dst;
    float cpu_dstScalar;
    float npp_res[4];
    Vec3i cpu_IdxScalar;
    Pixel32sC4 cpu_idxX;
    Pixel32sC4 cpu_idxY;
    Pixel32sC4 npp_idxX;
    Pixel32sC4 npp_idxY;
    nv::Image32fC4 npp_src1(size, size);
    mpp::cuda::DevVar<float> npp_dst(4);
    mpp::cuda::DevVar<int> npp_dstIdxX(4);
    mpp::cuda::DevVar<int> npp_dstIdxY(4);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSize(nppCtx));

    cpu_src1.Set(127);
    cpu_src1(10, 10)  = {200, 201, 202, 203};
    cpu_src1(100, 10) = {4, 5, 6, 7};
    cpu_src1(104, 10) = {4, 5, 6, 7};
    cpu_src1(10, 100) = {4, 5, 6, 7};
    cpu_src1(9, 120)  = {4, 5, 6, 7};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndx(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);
    CHECK(cpu_src1(cpu_idxX.w, cpu_idxY.w).w == 7);
    CHECK(cpu_idxX.w == 100);
    CHECK(cpu_idxY.w == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2], npp_res[3]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
    CHECK(cpu_dst.w == npp_res[3]);
}

TEST_CASE("32fC4A", "[NPP.Statistics.MinIndex]")
{
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::ImageView<Pixel32fC4A> cpu_src1A(cpu_src1);
    Pixel32fC4A cpu_dst;
    float cpu_dstScalar;
    float npp_res[3];
    Vec3i cpu_IdxScalar;
    Pixel32sC4A cpu_idxX;
    Pixel32sC4A cpu_idxY;
    Pixel32sC4A npp_idxX;
    Pixel32sC4A npp_idxY;
    nv::Image32fC4 npp_src1(size, size);
    mpp::cuda::DevVar<float> npp_dst(3);
    mpp::cuda::DevVar<int> npp_dstIdxX(3);
    mpp::cuda::DevVar<int> npp_dstIdxY(3);
    mpp::cuda::DevVar<byte> npp_buffer(npp_src1.MinIndxGetBufferHostSizeA(nppCtx));

    cpu_src1.Set(127);
    cpu_src1A(10, 10)  = {200, 201, 202};
    cpu_src1A(100, 10) = {4, 5, 6};
    cpu_src1A(104, 10) = {4, 5, 6};
    cpu_src1A(10, 100) = {4, 5, 6};
    cpu_src1A(9, 120)  = {4, 5, 6};

    cpu_src1 >> npp_src1;

    npp_src1.MinIndxA(npp_buffer, npp_dst, npp_dstIdxX, npp_dstIdxY, nppCtx);
    npp_dst >> npp_res;
    npp_dstIdxX >> npp_idxX.data();
    npp_dstIdxY >> npp_idxY.data();

    cpu_src1A.MinIndex(cpu_dst, cpu_idxX, cpu_idxY, cpu_dstScalar, cpu_IdxScalar);

    CHECK(cpu_src1(cpu_idxX.x, cpu_idxY.x).x == 4);
    CHECK(cpu_idxX.x == 100);
    CHECK(cpu_idxY.x == 10);
    CHECK(cpu_src1(cpu_idxX.y, cpu_idxY.y).y == 5);
    CHECK(cpu_idxX.y == 100);
    CHECK(cpu_idxY.y == 10);
    CHECK(cpu_src1(cpu_idxX.z, cpu_idxY.z).z == 6);
    CHECK(cpu_idxX.z == 100);
    CHECK(cpu_idxY.z == 10);

    // BUG in NPP function, the returned index is NOT the correct pixel coordinate!
    /*CHECK(cpu_idxX.x == npp_idxX.x);
    CHECK(cpu_idxY.x == npp_idxY.x);
    CHECK(cpu_src1(npp_idxX.x, npp_idxY.x) == 4);*/
    CHECK(cpu_dstScalar == std::min({npp_res[0], npp_res[1], npp_res[2]}));
    CHECK(cpu_dst.x == npp_res[0]);
    CHECK(cpu_dst.y == npp_res[1]);
    CHECK(cpu_dst.z == npp_res[2]);
}