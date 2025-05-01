#include <backends/npp/image/image16f.h>
#include <backends/npp/image/image16fC1View.h>
#include <backends/npp/image/image16fC2View.h>
#include <backends/npp/image/image16fC3View.h>
#include <backends/npp/image/image16fC4View.h>
#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC2View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image32fc.h>
#include <backends/npp/image/image32fcC1View.h>
#include <backends/npp/image/image32fcC2View.h>
#include <backends/npp/image/image32fcC3View.h>
#include <backends/npp/image/image32fcC4View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32sc.h>
#include <backends/npp/image/image32scC1View.h>
#include <backends/npp/image/image32scC2View.h>
#include <backends/npp/image/image32scC3View.h>
#include <backends/npp/image/image32scC4View.h>
#include <backends/npp/image/image8s.h>
#include <backends/npp/image/image8sC1View.h>
#include <backends/npp/image/image8sC2View.h>
#include <backends/npp/image/image8sC3View.h>
#include <backends/npp/image/image8sC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;
namespace cpu = opp::image::cpuSimple;
namespace nv  = opp::image::npp;

constexpr int size = 256;

TEST_CASE("Remap", "[NPP.Arithmetic.Abs]")
{
    std::vector<byte> ub = {
        0,   0,   128, 63,  5,   244, 127, 63,  25,  208, 127, 63,  65,  148, 127, 63,  138, 64,  127, 63,  2,   213,
        126, 63,  190, 81,  126, 63,  215, 182, 125, 63,  106, 4,   125, 63,  151, 58,  124, 63,  136, 89,  123, 63,
        99,  97,  122, 63,  89,  82,  121, 63,  155, 44,  120, 63,  99,  240, 118, 63,  234, 157, 117, 63,  113, 53,
        116, 63,  57,  183, 114, 63,  140, 35,  113, 63,  179, 122, 111, 63,  255, 188, 109, 63,  193, 234, 107, 63,
        83,  4,   106, 63,  10,  10,  104, 63,  74,  252, 101, 63,  111, 219, 99,  63,  224, 167, 97,  63,  5,   98,
        95,  63,  71,  10,  93,  63,  21,  161, 90,  63,  224, 38,  88,  63,  29,  156, 85,  63,  64,  1,   83,  63,
        196, 86,  80,  63,  37,  157, 77,  63,  222, 212, 74,  63,  115, 254, 71,  63,  101, 26,  69,  63,  56,  41,
        66,  63,  116, 43,  63,  63,  158, 33,  60,  63,  68,  12,  57,  63,  240, 235, 53,  63,  46,  193, 50,  63,
        142, 140, 47,  63,  157, 78,  44,  63,  238, 7,   41,  63,  16,  185, 37,  63,  151, 98,  34,  63,  20,  5,
        31,  63,  28,  161, 27,  63,  66,  55,  24,  63,  26,  200, 20,  63,  56,  84,  17,  63,  47,  220, 13,  63,
        149, 96,  10,  63,  253, 225, 6,   63,  249, 96,  3,   63,  59,  188, 255, 62,  245, 179, 248, 62,  69,  170,
        241, 62,  77,  160, 234, 62,  44,  151, 227, 62,  253, 143, 220, 62,  220, 139, 213, 62,  221, 139, 206, 62,
        19,  145, 199, 62,  151, 156, 192, 62,  116, 175, 185, 62,  176, 202, 178, 62,  83,  239, 171, 62,  98,  30,
        165, 62,  212, 88,  158, 62,  175, 159, 151, 62,  221, 243, 144, 62,  81,  86,  138, 62,  248, 199, 131, 62,
        110, 147, 122, 62,  222, 184, 109, 62,  233, 1,   97,  62,  77,  112, 84,  62,  159, 5,   72,  62,  116, 195,
        59,  62,  88,  171, 47,  62,  187, 190, 35,  62,  9,   255, 23,  62,  176, 109, 12,  62,  2,   12,  1,   62,
        119, 182, 235, 61,  52,  185, 213, 61,  144, 34,  192, 61,  199, 244, 170, 61,  249, 49,  150, 61,  37,  220,
        129, 61,  98,  234, 91,  61,  206, 253, 52,  61,  231, 245, 14,  61,  148, 171, 211, 60,  187, 64,  139, 60,
        29,  97,  9,   60,  0,   0,   0,   128, 28,  153, 5,   188, 58,  177, 131, 188, 122, 170, 194, 188, 61,  181,
        255, 188, 114, 103, 29,  189, 165, 250, 57,  189, 112, 147, 85,  189, 16,  49,  112, 189, 152, 233, 132, 189,
        201, 60,  145, 189, 43,  18,  157, 189, 213, 105, 168, 189, 4,   68,  179, 189, 9,   161, 189, 189, 88,  129,
        199, 189, 114, 229, 208, 189, 3,   206, 217, 189, 199, 59,  226, 189, 144, 47,  234, 189, 71,  170, 241, 189,
        242, 172, 248, 189, 176, 56,  255, 189, 88,  167, 2,   190, 28,  120, 5,   190, 87,  15,  8,   190, 191, 109,
        10,  190, 25,  148, 12,  190, 46,  131, 14,  190, 219, 59,  16,  190, 252, 190, 17,  190, 123, 13,  19,  190,
        73,  40,  20,  190, 95,  16,  21,  190, 190, 198, 21,  190, 109, 76,  22,  190, 125, 162, 22,  190, 4,   202,
        22,  190, 32,  196, 22,  190, 241, 145, 22,  190, 161, 52,  22,  190, 92,  173, 21,  190, 86,  253, 20,  190,
        194, 37,  20,  190, 221, 39,  19,  190, 230, 4,   18,  190, 30,  190, 16,  190, 201, 84,  15,  190, 47,  202,
        13,  190, 154, 31,  12,  190, 82,  86,  10,  190, 166, 111, 8,   190, 228, 108, 6,   190, 90,  79,  4,   190,
        86,  24,  2,   190, 78,  146, 255, 189, 55,  198, 250, 189, 254, 206, 245, 189, 68,  175, 240, 189, 154, 105,
        235, 189, 151, 0,   230, 189, 194, 118, 224, 189, 171, 206, 218, 189, 213, 10,  213, 189, 193, 45,  207, 189,
        231, 57,  201, 189, 177, 49,  195, 189, 138, 23,  189, 189, 214, 237, 182, 189, 229, 182, 176, 189, 17,  117,
        170, 189, 153, 42,  164, 189, 178, 217, 157, 189, 155, 132, 151, 189, 102, 45,  145, 189, 49,  214, 138, 189,
        9,   129, 132, 189, 221, 95,  124, 189, 164, 201, 111, 189, 54,  67,  99,  189, 65,  208, 86,  189, 88,  116,
        74,  189, 241, 50,  62,  189, 108, 15,  50,  189, 5,   13,  38,  189, 219, 46,  26,  189, 237, 119, 14,  189,
        31,  235, 2,   189, 105, 22,  239, 188, 169, 181, 216, 188, 8,   185, 194, 188, 77,  37,  173, 188, 5,   255,
        151, 188, 141, 74,  131, 188, 120, 23,  94,  188, 37,  141, 54,  188, 35,  253, 15,  188, 229, 219, 212, 187,
        101, 203, 139, 187, 10,  168, 9,   187, 0,   0,   0,   0,   96,  79,  5,   59,  81,  26,  131, 59,  46,  81,
        193, 59,  187, 69,  253, 59,  67,  121, 27,  60,  152, 41,  55,  60,  36,  178, 81,  60,  145, 17,  107, 60,
        132, 163, 129, 60,  14,  41,  141, 60,  103, 25,  152, 60,  207, 116, 162, 60,  184, 59,  172, 60,  183, 110,
        181, 60,  158, 14,  190, 60,  127, 28,  198, 60,  140, 153, 205, 60,  43,  135, 212, 60,  236, 230, 218, 60,
        128, 186, 224, 60,  230, 3,   230, 60,  38,  197, 234, 60,  110, 0,   239, 60,  41,  184, 242, 60,  202, 238,
        245, 60,  251, 166, 248, 60,  140, 227, 250, 60,  69,  167, 252, 60,  68,  245, 253, 60,  139, 208, 254, 60,
        91,  60,  255, 60,  11,  60,  255, 60,  244, 210, 254, 60,  152, 4,   254, 60,  119, 212, 252, 60,  50,  70,
        251, 60,  132, 93,  249, 60,  20,  30,  247, 60,  190, 139, 244, 60,  66,  170, 241, 60,  140, 125, 238, 60,
        105, 9,   235, 60,  206, 81,  231, 60,  168, 90,  227, 60,  217, 39,  223, 60,  86,  189, 218, 60,  22,  31,
        214, 60,  235, 80,  209, 60,  208, 86,  204, 60,  162, 52,  199, 60,  50,  238, 193, 60,  96,  135, 188, 60,
        232, 3,   183, 60,  140, 103, 177, 60,  1,   182, 171, 60,  222, 242, 165, 60,  201, 33,  160, 60,  61,  70,
        154, 60,  175, 99,  148, 60,  119, 125, 142, 60,  237, 150, 136, 60,  78,  179, 130, 60,  82,  171, 121, 60,
        82,  2,   110, 60,  122, 113, 98,  60,  127, 254, 86,  60,  235, 174, 75,  60,  38,  136, 64,  60,  78,  143,
        53,  60,  87,  201, 42,  60,  9,   59,  32,  60,  213, 232, 21,  60,  25,  215, 11,  60,  219, 9,   2,   60,
        238, 9,   241, 59,  60,  152, 222, 59,  64,  197, 204, 59,  112, 151, 187, 59,  129, 20,  171, 59,  251, 65,
        155, 59,  188, 36,  140, 59,  109, 130, 123, 59,  183, 54,  96,  59,  184, 109, 70,  59,  79,  45,  46,  59,
        219, 122, 23,  59,  56,  90,  2,   59,  129, 158, 221, 58,  72,  184, 185, 58,  117, 5,   153, 58,  20,  14,
        119, 58,  98,  121, 66,  58,  121, 73,  20,  58,  237, 237, 216, 57,  231, 236, 149, 57,  186, 236, 62,  57,
        172, 157, 213, 56,  31,  201, 60,  56,  188, 145, 59,  55};

    double *ttt = reinterpret_cast<double *>(ub.data());

    if (ttt[0] > 0)
    {
    }

    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_dst(5, 1);
    cpu::Image<Pixel32fC1> npp_res(5, 1);
    nv::Image32fC1 npp_src1(6, 6);
    nv::Image32fC1 npp_dst(5, 1);
    nv::Image32fC1 npp_X(5, 1);
    nv::Image32fC1 npp_Y(5, 1);

    std::vector<Pixel32fC1> srcImg = {0.5688f, 0.4694f, 0.0119f, 0.3371f, 0.1622f, 0.7943f, 0.3112f, 0.5285f, 0.1656f,
                                      0.6020f, 0.2630f, 0.6541f, 0.6892f, 0.7482f, 0.4505f, 0.0838f, 0.2290f, 0.9133f,
                                      0.1524f, 0.8258f, 0.5383f, 0.9961f, 0.0782f, 0.4427f, 0.1067f, 0.9619f, 0.0046f,
                                      0.7749f, 0.8173f, 0.8687f, 0.0844f, 0.3998f, 0.2599f, 0.8001f, 0.4314f, 0.9106f};

    std::vector<Pixel32fC1> checkpointsX = {{2.4f}, {2.2f}, {2.5f}, {2.11f}, {2.9f}};
    std::vector<Pixel32fC1> checkpointsY = {{2.1f}, {2.3f}, {2.5f}, {2.64f}, {2.8f}};

    npp_X << checkpointsX;
    npp_Y << checkpointsY;

    npp_src1 << srcImg;

    npp_src1.Remap(npp_X, npp_Y, npp_dst, NPPI_INTER_LANCZOS, nppCtx);

    npp_res << npp_dst;

    std::cout << std::setprecision(10) << npp_res(0, 0).x << "f," << std::endl;
    std::cout << std::setprecision(10) << npp_res(1, 0).x << "f," << std::endl;
    std::cout << std::setprecision(10) << npp_res(2, 0).x << "f," << std::endl;
    std::cout << std::setprecision(10) << npp_res(3, 0).x << "f," << std::endl;
    std::cout << std::setprecision(10) << npp_res(4, 0).x << "f," << std::endl;
}

TEST_CASE("16sC1", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC1::GetStreamContext();

    cpu::Image<Pixel16sC1> cpu_src1(size, size);
    cpu::Image<Pixel16sC1> cpu_dst(size, size);
    cpu::Image<Pixel16sC1> npp_res(size, size);
    nv::Image16sC1 npp_src1(size, size);
    nv::Image16sC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);
    // make sure that this special case occurs (-32768 -> 32767)
    cpu_src1(0, 0) = numeric_limits<short>::min();

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16sC3", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC3::GetStreamContext();

    cpu::Image<Pixel16sC3> cpu_src1(size, size);
    cpu::Image<Pixel16sC3> cpu_dst(size, size);
    cpu::Image<Pixel16sC3> npp_res(size, size);
    nv::Image16sC3 npp_src1(size, size);
    nv::Image16sC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16sC4", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16sC4A", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16sC4::GetStreamContext();

    cpu::Image<Pixel16sC4> cpu_src1(size, size);
    cpu::Image<Pixel16sC4> cpu_dst(size, size);
    cpu::Image<Pixel16sC4> npp_res(size, size);
    nv::Image16sC4 npp_src1(size, size);
    nv::Image16sC4 npp_dst(size, size);
    cpu::ImageView<Pixel16sC4A> cpu_src1A = cpu_src1;
    cpu::ImageView<Pixel16sC4A> cpu_dstA  = cpu_dst;

    cpu_dst.Set({127, 127, 127, 127});
    npp_dst.Set({127, 127, 127, 127}, nppCtx);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1A.Abs(cpu_dstA);
    npp_src1.AbsA(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Abs();
    npp_src1.AbsA(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16fC1", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC1::GetStreamContext();

    cpu::Image<Pixel16fC1> cpu_src1(size, size);
    cpu::Image<Pixel16fC1> cpu_dst(size, size);
    cpu::Image<Pixel16fC1> npp_res(size, size);
    nv::Image16fC1 npp_src1(size, size);
    nv::Image16fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16fC3", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC3::GetStreamContext();

    cpu::Image<Pixel16fC3> cpu_src1(size, size);
    cpu::Image<Pixel16fC3> cpu_dst(size, size);
    cpu::Image<Pixel16fC3> npp_res(size, size);
    nv::Image16fC3 npp_src1(size, size);
    nv::Image16fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("16fC4", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image16fC4::GetStreamContext();

    cpu::Image<Pixel16fC4> cpu_src1(size, size);
    cpu::Image<Pixel16fC4> cpu_dst(size, size);
    cpu::Image<Pixel16fC4> npp_res(size, size);
    nv::Image16fC4 npp_src1(size, size);
    nv::Image16fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("32fC1", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC1::GetStreamContext();

    cpu::Image<Pixel32fC1> cpu_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst(size, size);
    cpu::Image<Pixel32fC1> npp_res(size, size);
    nv::Image32fC1 npp_src1(size, size);
    nv::Image32fC1 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("32fC3", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC3::GetStreamContext();

    cpu::Image<Pixel32fC3> cpu_src1(size, size);
    cpu::Image<Pixel32fC3> cpu_dst(size, size);
    cpu::Image<Pixel32fC3> npp_res(size, size);
    nv::Image32fC3 npp_src1(size, size);
    nv::Image32fC3 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("32fC4", "[NPP.Arithmetic.Abs]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image32fC4::GetStreamContext();

    cpu::Image<Pixel32fC4> cpu_src1(size, size);
    cpu::Image<Pixel32fC4> cpu_dst(size, size);
    cpu::Image<Pixel32fC4> npp_res(size, size);
    nv::Image32fC4 npp_src1(size, size);
    nv::Image32fC4 npp_dst(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    cpu_src1.Abs(cpu_dst);
    npp_src1.Abs(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1.Abs();
    npp_src1.Abs(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}

TEST_CASE("32fC4A", "[NPP.Arithmetic.Abs]")
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

    cpu_src1 >> npp_src1;

    cpu_src1A.Abs(cpu_dstA);
    npp_src1.AbsA(npp_dst, nppCtx);

    npp_res << npp_dst;

    CHECK(cpu_dst.IsIdentical(npp_res));

    cpu_src1A.Abs();
    npp_src1.AbsA(nppCtx);

    npp_res << npp_src1;

    CHECK(cpu_src1.IsIdentical(npp_res));
}