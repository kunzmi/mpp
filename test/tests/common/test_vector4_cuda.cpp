#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVar.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/limits.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <vector>

using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace Catch;

namespace opp::cuda
{
void runtest_vector4_kernel(Vector4<byte> *aDataByte, Vector4<byte> *aCompByte,         //
                            Vector4<sbyte> *resSBytes, Vector4<byte> *aCompSbyte,       //
                            Vector4<short> *aDataShort, Vector4<byte> *aCompShort,      //
                            Vector4<ushort> *aDataUShort, Vector4<byte> *aCompUShort,   //
                            Vector4<BFloat16> *aDataBFloat, Vector4<byte> *aCompBFloat, //
                            Vector4<HalfFp16> *aDataHalf, Vector4<byte> *aCompHalf,     //
                            Vector4<float> *aDataFloat, Vector4<byte> *aCompFloat);
}

TEST_CASE("Vector<short> SIMD CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    DevVar<Vector4<sbyte>> sbytes(37);
    DevVar<Vector4<byte>> compSBytes(37);
    DevVar<Vector4<byte>> bytes(37);
    DevVar<Vector4<byte>> compBytes(37);
    DevVar<Vector4<short>> shorts(37);
    DevVar<Vector4<byte>> compShorts(37);
    DevVar<Vector4<ushort>> ushorts(37);
    DevVar<Vector4<byte>> compUShorts(37);
    DevVar<Vector4<BFloat16>> bfloats(37);
    DevVar<Vector4<byte>> compBFloats(37);
    DevVar<Vector4<HalfFp16>> halfs(37);
    DevVar<Vector4<byte>> compHalfs(37);
    DevVar<Vector4<float>> floats(37);
    DevVar<Vector4<byte>> compFloats(37);

    runtest_vector4_kernel(bytes.Pointer(), compBytes.Pointer(),     //
                           sbytes.Pointer(), compSBytes.Pointer(),   //
                           shorts.Pointer(), compShorts.Pointer(),   //
                           ushorts.Pointer(), compUShorts.Pointer(), //
                           bfloats.Pointer(), compBFloats.Pointer(), //
                           halfs.Pointer(), compHalfs.Pointer(),     //
                           floats.Pointer(), compFloats.Pointer());

    cudaSafeCall(cudaDeviceSynchronize());
    std::vector<Vector4<sbyte>> resSBytes(37);
    std::vector<Vector4<byte>> resCompSBytes(37);
    std::vector<Vector4<byte>> resBytes(37);
    std::vector<Vector4<byte>> resCompBytes(37);
    std::vector<Vector4<short>> resShort(37);
    std::vector<Vector4<byte>> resCompShort(37);
    std::vector<Vector4<ushort>> resUShort(37);
    std::vector<Vector4<byte>> resCompUShort(37);
    std::vector<Vector4<BFloat16>> resBFloat(37);
    std::vector<Vector4<byte>> resCompBFloat(37);
    std::vector<Vector4<HalfFp16>> resHalf(37);
    std::vector<Vector4<byte>> resCompHalf(37);
    std::vector<Vector4<float>> resFloat(37);
    std::vector<Vector4<byte>> resCompFloat(37);

    sbytes >> resSBytes;
    compSBytes >> resCompSBytes;

    bytes >> resBytes;
    compBytes >> resCompBytes;

    shorts >> resShort;
    compShorts >> resCompShort;

    ushorts >> resUShort;
    compUShorts >> resCompUShort;

    bfloats >> resBFloat;
    compBFloats >> resCompBFloat;

    halfs >> resHalf;
    compHalfs >> resCompHalf;

    floats >> resFloat;
    compFloats >> resCompFloat;

    // sbyte
    Vector4<sbyte> sbA(12, -120, -10, 120);
    Vector4<sbyte> sbB(120, -80, 30, 5);
    Vector4<sbyte> sbC(12, 20, 0, 120);

    CHECK(resSBytes[0] == Vector4<sbyte>(Vector4<float>(sbA) + Vector4<float>(sbB)));
    CHECK(resSBytes[1] == Vector4<sbyte>(Vector4<float>(sbA) - Vector4<float>(sbB)));
    CHECK(resSBytes[2] == sbA * sbB);
    CHECK(resSBytes[3] == sbA / sbB);

    CHECK(resSBytes[4] == Vector4<sbyte>(Vector4<float>(sbA) + Vector4<float>(sbB)));
    CHECK(resSBytes[5] == Vector4<sbyte>(Vector4<float>(sbA) - Vector4<float>(sbB)));
    CHECK(resSBytes[6] == sbA * sbB);
    CHECK(resSBytes[7] == sbA / sbB);

    CHECK(resSBytes[8] == Vector4<sbyte>(-12, 120, 10, -120));
    CHECK(resSBytes[12] == Vector4<sbyte>(12, 120, 10, 120));
    CHECK(resSBytes[13] == Vector4<sbyte>(108, 40, 40, 115));
    CHECK(resSBytes[17] == Vector4<sbyte>(12, 120, 10, 120));
    CHECK(resSBytes[18] == Vector4<sbyte>(108, 40, 40, 115));

    CHECK(resSBytes[19] == Vector4<sbyte>(12, -120, -10, 5));
    CHECK(resSBytes[20] == Vector4<sbyte>(120, -80, 30, 120));

    CHECK(resSBytes[21] == Vector4<sbyte>(12, -120, -10, 5));
    CHECK(resSBytes[22] == Vector4<sbyte>(120, -80, 30, 120));

    CHECK(resSBytes[35] == Vector4<sbyte>(Vector4<float>(sbB) - Vector4<float>(sbA)));
    CHECK(resSBytes[36] == sbB / sbA);

    CHECK(resCompSBytes[0] == Vector4<sbyte>::CompareEQ(sbA, sbC));
    CHECK(resCompSBytes[1] == Vector4<sbyte>::CompareLE(sbA, sbC));
    CHECK(resCompSBytes[2] == Vector4<sbyte>::CompareLT(sbA, sbC));
    CHECK(resCompSBytes[3] == Vector4<sbyte>::CompareGE(sbA, sbC));
    CHECK(resCompSBytes[4] == Vector4<sbyte>::CompareGT(sbA, sbC));
    CHECK(resCompSBytes[5] == Vector4<sbyte>::CompareNEQ(sbA, sbC));

    // ==
    CHECK(resCompSBytes[6] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[7] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[8] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[9] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[10] == Pixel8uC4(byte(0)));
    // <=
    CHECK(resCompSBytes[11] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[12] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[13] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[14] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[15] == Pixel8uC4(byte(0)));
    // <
    CHECK(resCompSBytes[16] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[17] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[18] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[19] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[20] == Pixel8uC4(byte(0)));
    // >=
    CHECK(resCompSBytes[21] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[22] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[23] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[24] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[25] == Pixel8uC4(byte(0)));
    // >
    CHECK(resCompSBytes[26] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[27] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[28] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[29] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[30] == Pixel8uC4(byte(0)));
    // !=
    CHECK(resCompSBytes[31] == Pixel8uC4(byte(0)));
    CHECK(resCompSBytes[32] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[33] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[34] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[35] == Pixel8uC4(byte(1)));
    CHECK(resCompSBytes[36] == Pixel8uC4(byte(1)));

    // byte
    Vector4<byte> bA(12, 120, 100, 120);
    Vector4<byte> bB(120, 180, 30, 5);
    Vector4<byte> bC(12, 20, 100, 120);

    CHECK(resBytes[0] == Vector4<byte>(Vector4<float>(bA) + Vector4<float>(bB)));
    CHECK(resBytes[1] == Vector4<byte>(Vector4<float>(bA) - Vector4<float>(bB)));
    CHECK(resBytes[2] == bA * bB);
    CHECK(resBytes[3] == bA / bB);

    CHECK(resBytes[4] == Vector4<byte>(Vector4<float>(bA) + Vector4<float>(bB)));
    CHECK(resBytes[5] == Vector4<byte>(Vector4<float>(bA) - Vector4<float>(bB)));
    CHECK(resBytes[6] == bA * bB);
    CHECK(resBytes[7] == bA / bB);

    CHECK(resBytes[13] == Vector4<byte>(108, 60, 70, 115));
    CHECK(resBytes[18] == Vector4<byte>(108, 60, 70, 115));

    CHECK(resBytes[19] == Vector4<byte>(12, 120, 30, 5));
    CHECK(resBytes[20] == Vector4<byte>(120, 180, 100, 120));
    CHECK(resBytes[21] == Vector4<byte>(12, 120, 30, 5));
    CHECK(resBytes[22] == Vector4<byte>(120, 180, 100, 120));

    CHECK(resBytes[35] == Vector4<byte>(Vector4<float>(bB) - Vector4<float>(bA)));
    CHECK(resBytes[36] == bB / bA);

    CHECK(resCompBytes[0] == Vector4<byte>::CompareEQ(bA, bC));
    CHECK(resCompBytes[1] == Vector4<byte>::CompareLE(bA, bC));
    CHECK(resCompBytes[2] == Vector4<byte>::CompareLT(bA, bC));
    CHECK(resCompBytes[3] == Vector4<byte>::CompareGE(bA, bC));
    CHECK(resCompBytes[4] == Vector4<byte>::CompareGT(bA, bC));
    CHECK(resCompBytes[5] == Vector4<byte>::CompareNEQ(bA, bC));

    // ==
    CHECK(resCompBytes[6] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[7] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[8] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[9] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[10] == Pixel8uC4(byte(0)));
    // <=
    CHECK(resCompBytes[11] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[12] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[13] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[14] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[15] == Pixel8uC4(byte(0)));
    // <
    CHECK(resCompBytes[16] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[17] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[18] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[19] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[20] == Pixel8uC4(byte(0)));
    // >=
    CHECK(resCompBytes[21] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[22] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[23] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[24] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[25] == Pixel8uC4(byte(0)));
    // >
    CHECK(resCompBytes[26] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[27] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[28] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[29] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[30] == Pixel8uC4(byte(0)));
    // !=
    CHECK(resCompBytes[31] == Pixel8uC4(byte(0)));
    CHECK(resCompBytes[32] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[33] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[34] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[35] == Pixel8uC4(byte(1)));
    CHECK(resCompBytes[36] == Pixel8uC4(byte(1)));

    // short
    Vector4<short> sA(12, -30000, -10, 31000);
    Vector4<short> sB(120, -3000, 30, -4096);
    Vector4<short> sC(12, -30000, 0, 31000);

    CHECK(resShort[0] == Vector4<short>(Vector4<float>(sA) + Vector4<float>(sB)));
    CHECK(resShort[1] == Vector4<short>(Vector4<float>(sA) - Vector4<float>(sB)));
    CHECK(resShort[2] == sA * sB);
    CHECK(resShort[3] == sA / sB);

    CHECK(resShort[4] == Vector4<short>(Vector4<float>(sA) + Vector4<float>(sB)));
    CHECK(resShort[5] == Vector4<short>(Vector4<float>(sA) - Vector4<float>(sB)));
    CHECK(resShort[6] == sA * sB);
    CHECK(resShort[7] == sA / sB);

    CHECK(resShort[8] == Vector4<short>(-12, 30000, 10, -31000));
    CHECK(resShort[12] == Vector4<short>(12, 30000, 10, 31000));
    CHECK(resShort[13] == Vector4<short>(108, 27000, 40, -30440));
    CHECK(resShort[17] == Vector4<short>(12, 30000, 10, 31000));
    CHECK(resShort[18] == Vector4<short>(108, 27000, 40, -30440));

    CHECK(resShort[19] == Vector4<short>(12, -30000, -10, -4096));
    CHECK(resShort[20] == Vector4<short>(120, -3000, 30, 31000));

    CHECK(resShort[21] == Vector4<short>(12, -30000, -10, -4096));
    CHECK(resShort[22] == Vector4<short>(120, -3000, 30, 31000));

    CHECK(resShort[35] == Vector4<short>(Vector4<float>(sB) - Vector4<float>(sA)));
    CHECK(resShort[36] == sB / sA);

    CHECK(resCompShort[0] == Vector4<short>::CompareEQ(sA, sC));
    CHECK(resCompShort[1] == Vector4<short>::CompareLE(sA, sC));
    CHECK(resCompShort[2] == Vector4<short>::CompareLT(sA, sC));
    CHECK(resCompShort[3] == Vector4<short>::CompareGE(sA, sC));
    CHECK(resCompShort[4] == Vector4<short>::CompareGT(sA, sC));
    CHECK(resCompShort[5] == Vector4<short>::CompareNEQ(sA, sC));

    // ==
    CHECK(resCompShort[6] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[7] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[8] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[9] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[10] == Pixel8uC4(byte(0)));
    // <=
    CHECK(resCompShort[11] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[12] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[13] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[14] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[15] == Pixel8uC4(byte(0)));
    // <
    CHECK(resCompShort[16] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[17] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[18] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[19] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[20] == Pixel8uC4(byte(0)));
    // >=
    CHECK(resCompShort[21] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[22] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[23] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[24] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[25] == Pixel8uC4(byte(0)));
    // >
    CHECK(resCompShort[26] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[27] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[28] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[29] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[30] == Pixel8uC4(byte(0)));
    // !=
    CHECK(resCompShort[31] == Pixel8uC4(byte(0)));
    CHECK(resCompShort[32] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[33] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[34] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[35] == Pixel8uC4(byte(1)));
    CHECK(resCompShort[36] == Pixel8uC4(byte(1)));

    // ushort
    Vector4<ushort> usA(12, 60000, 100, 120);
    Vector4<ushort> usB(120, 7000, 30, 5);
    Vector4<ushort> usC(12, 20, 100, 120);

    CHECK(resUShort[0] == Vector4<ushort>(Vector4<float>(usA) + Vector4<float>(usB)));
    CHECK(resUShort[1] == Vector4<ushort>(Vector4<float>(usA) - Vector4<float>(usB)));
    CHECK(resUShort[2] == usA * usB);
    CHECK(resUShort[3] == usA / usB);

    CHECK(resUShort[4] == Vector4<ushort>(Vector4<float>(usA) + Vector4<float>(usB)));
    CHECK(resUShort[5] == Vector4<ushort>(Vector4<float>(usA) - Vector4<float>(usB)));
    CHECK(resUShort[6] == usA * usB);
    CHECK(resUShort[7] == usA / usB);

    CHECK(resUShort[13] == Vector4<ushort>(108, 53000, 70, 115));
    CHECK(resUShort[18] == Vector4<ushort>(108, 53000, 70, 115));

    CHECK(resUShort[19] == Vector4<ushort>(12, 7000, 30, 5));
    CHECK(resUShort[20] == Vector4<ushort>(120, 60000, 100, 120));
    CHECK(resUShort[21] == Vector4<ushort>(12, 7000, 30, 5));
    CHECK(resUShort[22] == Vector4<ushort>(120, 60000, 100, 120));

    CHECK(resUShort[35] == Vector4<ushort>(Vector4<float>(usB) - Vector4<float>(usA)));
    CHECK(resUShort[36] == usB / usA);

    CHECK(resCompUShort[0] == Vector4<ushort>::CompareEQ(usA, usC));
    CHECK(resCompUShort[1] == Vector4<ushort>::CompareLE(usA, usC));
    CHECK(resCompUShort[2] == Vector4<ushort>::CompareLT(usA, usC));
    CHECK(resCompUShort[3] == Vector4<ushort>::CompareGE(usA, usC));
    CHECK(resCompUShort[4] == Vector4<ushort>::CompareGT(usA, usC));
    CHECK(resCompUShort[5] == Vector4<ushort>::CompareNEQ(usA, usC));

    // ==
    CHECK(resCompUShort[6] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[7] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[8] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[9] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[10] == Pixel8uC4(byte(0)));
    // <=
    CHECK(resCompUShort[11] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[12] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[13] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[14] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[15] == Pixel8uC4(byte(0)));
    // <
    CHECK(resCompUShort[16] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[17] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[18] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[19] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[20] == Pixel8uC4(byte(0)));
    // >=
    CHECK(resCompUShort[21] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[22] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[23] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[24] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[25] == Pixel8uC4(byte(0)));
    // >
    CHECK(resCompUShort[26] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[27] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[28] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[29] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[30] == Pixel8uC4(byte(0)));
    // !=
    CHECK(resCompUShort[31] == Pixel8uC4(byte(0)));
    CHECK(resCompUShort[32] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[33] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[34] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[35] == Pixel8uC4(byte(1)));
    CHECK(resCompUShort[36] == Pixel8uC4(byte(1)));

    // BFloat16
    Vector4<BFloat16> bfA(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f), BFloat16(310.12f));
    Vector4<BFloat16> bfB(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f), BFloat16(-4096.9f));
    Vector4<BFloat16> bfC(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(0.0f), BFloat16(310.12f));

    CHECK(resBFloat[0] == bfA + bfB);
    CHECK(resBFloat[1] == bfA - bfB);
    CHECK(resBFloat[2] == bfA * bfB);
    CHECK(resBFloat[3] == bfA / bfB);

    CHECK(resBFloat[4] == bfA + bfB);
    CHECK(resBFloat[5] == bfA - bfB);
    CHECK(resBFloat[6] == bfA * bfB);
    CHECK(resBFloat[7] == bfA / bfB);

    CHECK(resBFloat[8] == Vector4<BFloat16>(BFloat16(-12.4f), BFloat16(30000.2f), BFloat16(10.5f), BFloat16(-310.12f)));
    CHECK(resBFloat[9] == Vector4<BFloat16>::Exp(bfC));

    CHECK(resBFloat[10].x == Vector4<BFloat16>::Ln(bfC).x);
    CHECK(isnan(resBFloat[10].y));
    CHECK(resBFloat[10].z == Vector4<BFloat16>::Ln(bfC).z);
    CHECK(resBFloat[10].w == Vector4<BFloat16>::Ln(bfC).w);

    CHECK(resBFloat[11].x == Vector4<BFloat16>::Sqrt(bfC).x);
    CHECK(isnan(resBFloat[11].y));
    CHECK(resBFloat[11].z == Vector4<BFloat16>::Sqrt(bfC).z);
    CHECK(resBFloat[11].w == Vector4<BFloat16>::Sqrt(bfC).w);

    CHECK(resBFloat[12] == Vector4<BFloat16>(BFloat16(12.4f), BFloat16(30000.2f), BFloat16(10.5f), BFloat16(310.12f)));
    CHECK(resBFloat[17] == Vector4<BFloat16>(BFloat16(12.4f), BFloat16(30000.2f), BFloat16(10.5f), BFloat16(310.12f)));

    CHECK(resBFloat[19] ==
          Vector4<BFloat16>(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f), BFloat16(-4096.9f)));
    CHECK(resBFloat[20] == Vector4<BFloat16>(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f), BFloat16(310.12f)));

    CHECK(resBFloat[21] ==
          Vector4<BFloat16>(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f), BFloat16(-4096.9f)));
    CHECK(resBFloat[22] == Vector4<BFloat16>(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f), BFloat16(310.12f)));

    CHECK(resBFloat[23] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f), BFloat16(310.0f)));
    CHECK(resBFloat[24] == Vector4<BFloat16>(BFloat16(13.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));
    CHECK(resBFloat[25] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f), BFloat16(310.0f)));
    CHECK(resBFloat[26] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));
    CHECK(resBFloat[27] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));

    CHECK(resBFloat[28] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f), BFloat16(310.0f)));
    CHECK(resBFloat[29] == Vector4<BFloat16>(BFloat16(13.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));
    CHECK(resBFloat[30] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f), BFloat16(310.0f)));
    CHECK(resBFloat[31] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));
    CHECK(resBFloat[32] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));

    CHECK(resBFloat[33] == Vector4<BFloat16>(BFloat16(1.0f), BFloat16(2.0f), BFloat16(3.0f), BFloat16(4.0f)));
    CHECK(resBFloat[34] == Vector4<BFloat16>(BFloat16(2.0f), BFloat16(4.0f), BFloat16(6.0f), BFloat16(8.0f)));

    CHECK(resBFloat[35] == bfB - bfA);
    CHECK(resBFloat[36] == bfB / bfA);

    CHECK(resCompBFloat[0] == Vector4<BFloat16>::CompareEQ(bfA, bfC));
    CHECK(resCompBFloat[1] == Vector4<BFloat16>::CompareLE(bfA, bfC));
    CHECK(resCompBFloat[2] == Vector4<BFloat16>::CompareLT(bfA, bfC));
    CHECK(resCompBFloat[3] == Vector4<BFloat16>::CompareGE(bfA, bfC));
    CHECK(resCompBFloat[4] == Vector4<BFloat16>::CompareGT(bfA, bfC));
    CHECK(resCompBFloat[5] == Vector4<BFloat16>::CompareNEQ(bfA, bfC));

    // ==
    CHECK(resCompBFloat[6] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[7] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[8] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[9] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[10] == Pixel8uC4(byte(0)));
    // <=
    CHECK(resCompBFloat[11] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[12] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[13] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[14] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[15] == Pixel8uC4(byte(0)));
    // <
    CHECK(resCompBFloat[16] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[17] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[18] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[19] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[20] == Pixel8uC4(byte(0)));
    // >=
    CHECK(resCompBFloat[21] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[22] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[23] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[24] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[25] == Pixel8uC4(byte(0)));
    // >
    CHECK(resCompBFloat[26] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[27] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[28] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[29] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[30] == Pixel8uC4(byte(0)));
    // !=
    CHECK(resCompBFloat[31] == Pixel8uC4(byte(0)));
    CHECK(resCompBFloat[32] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[33] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[34] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[35] == Pixel8uC4(byte(1)));
    CHECK(resCompBFloat[36] == Pixel8uC4(byte(1)));

    // Half
    Vector4<HalfFp16> hA(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f), HalfFp16(310.12f));
    Vector4<HalfFp16> hB(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f), HalfFp16(-4096.9f));
    Vector4<HalfFp16> hC(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(0.0f), HalfFp16(310.12f));

    CHECK(resHalf[0] == hA + hB);
    CHECK(resHalf[1] == hA - hB);
    CHECK(resHalf[2] == hA * hB);
    CHECK(resHalf[3] == hA / hB);

    CHECK(resHalf[4] == hA + hB);
    CHECK(resHalf[5] == hA - hB);
    CHECK(resHalf[6] == hA * hB);
    CHECK(resHalf[7] == hA / hB);

    CHECK(resHalf[8] == Vector4<HalfFp16>(HalfFp16(-12.4f), HalfFp16(30000.2f), HalfFp16(10.5f), HalfFp16(-310.12f)));
    CHECK(resHalf[9] == Vector4<HalfFp16>::Exp(hC));

    CHECK(resHalf[10].x == Vector4<HalfFp16>::Ln(hC).x);
    CHECK(isnan(resHalf[10].y));
    CHECK(resHalf[10].z == Vector4<HalfFp16>::Ln(hC).z);
    CHECK(resHalf[10].w == Vector4<HalfFp16>::Ln(hC).w);

    CHECK(resHalf[11].x == Vector4<HalfFp16>::Sqrt(hC).x);
    CHECK(isnan(resBFloat[11].y));
    CHECK(resHalf[11].z == Vector4<HalfFp16>::Sqrt(hC).z);
    CHECK(resHalf[11].w == Vector4<HalfFp16>::Sqrt(hC).w);

    CHECK(resHalf[12] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(30000.2f), HalfFp16(10.5f), HalfFp16(310.12f)));
    CHECK(resHalf[17] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(30000.2f), HalfFp16(10.5f), HalfFp16(310.12f)));

    CHECK(resHalf[19] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f), HalfFp16(-4096.9f)));
    CHECK(resHalf[20] == Vector4<HalfFp16>(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f), HalfFp16(310.12f)));

    CHECK(resHalf[21] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f), HalfFp16(-4096.9f)));
    CHECK(resHalf[22] == Vector4<HalfFp16>(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f), HalfFp16(310.12f)));

    CHECK(resHalf[23] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    CHECK(resHalf[24] == Vector4<HalfFp16>(HalfFp16(13), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    CHECK(resHalf[25] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    CHECK(resHalf[26] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    CHECK(resHalf[27] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));

    CHECK(resHalf[28] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    CHECK(resHalf[29] == Vector4<HalfFp16>(HalfFp16(13), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    CHECK(resHalf[30] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    CHECK(resHalf[31] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    CHECK(resHalf[32] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));

    CHECK(resHalf[33] == Vector4<HalfFp16>(HalfFp16(1), HalfFp16(2), HalfFp16(3), HalfFp16(4)));
    CHECK(resHalf[34] == Vector4<HalfFp16>(HalfFp16(2), HalfFp16(4), HalfFp16(6), HalfFp16(8)));

    CHECK(resHalf[35] == hB - hA);
    CHECK(resHalf[36] == hB / hA);

    CHECK(resCompHalf[0] == Vector4<HalfFp16>::CompareEQ(hA, hC));
    CHECK(resCompHalf[1] == Vector4<HalfFp16>::CompareLE(hA, hC));
    CHECK(resCompHalf[2] == Vector4<HalfFp16>::CompareLT(hA, hC));
    CHECK(resCompHalf[3] == Vector4<HalfFp16>::CompareGE(hA, hC));
    CHECK(resCompHalf[4] == Vector4<HalfFp16>::CompareGT(hA, hC));
    CHECK(resCompHalf[5] == Vector4<HalfFp16>::CompareNEQ(hA, hC));

    // ==
    CHECK(resCompHalf[6] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[7] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[8] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[9] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[10] == Pixel8uC4(byte(0)));
    // <=
    CHECK(resCompHalf[11] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[12] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[13] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[14] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[15] == Pixel8uC4(byte(0)));
    // <
    CHECK(resCompHalf[16] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[17] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[18] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[19] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[20] == Pixel8uC4(byte(0)));
    // >=
    CHECK(resCompHalf[21] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[22] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[23] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[24] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[25] == Pixel8uC4(byte(0)));
    // >
    CHECK(resCompHalf[26] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[27] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[28] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[29] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[30] == Pixel8uC4(byte(0)));
    // !=
    CHECK(resCompHalf[31] == Pixel8uC4(byte(0)));
    CHECK(resCompHalf[32] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[33] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[34] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[35] == Pixel8uC4(byte(1)));
    CHECK(resCompHalf[36] == Pixel8uC4(byte(1)));

    // Float (32bit)
    Vector4<float> fA(12.4f, -30000.2f, -10.5f, 310.12f);
    Vector4<float> fB(120.1f, -3000.1f, 30.2f, -4096.9f);
    Vector4<float> fC(12.4f, -30000.2f, 0.0f, 310.12f);

    CHECK(resFloat[0] == fA + fB);
    CHECK(resFloat[1] == fA - fB);
    CHECK(resFloat[2] == fA * fB);
    CHECK(resFloat[3] == fA / fB);

    CHECK(resFloat[4] == fA + fB);
    CHECK(resFloat[5] == fA - fB);
    CHECK(resFloat[6] == fA * fB);
    CHECK(resFloat[7] == fA / fB);

    CHECK(resFloat[8] == Vector4<float>(-12.4f, 30000.2f, 10.5f, -310.12f));
    // GCC slighlty differs for exact comparison:
    CHECK(resFloat[9].x == Approx(Vector4<float>::Exp(fC).x).margin(0.001));
    CHECK(resFloat[9].y == Vector4<float>::Exp(fC).y);
    CHECK(resFloat[9].z == Vector4<float>::Exp(fC).z);
    CHECK(resFloat[9].w == Vector4<float>::Exp(fC).w);

    CHECK(resFloat[10].x == Vector4<float>::Ln(fC).x);
    CHECK(isnan(resFloat[10].y));
    CHECK(resFloat[10].z == Vector4<float>::Ln(fC).z);
    CHECK(resFloat[10].w == Vector4<float>::Ln(fC).w);

    CHECK(resFloat[11].x == Vector4<float>::Sqrt(fC).x);
    CHECK(isnan(resFloat[11].y));
    CHECK(resFloat[11].z == Vector4<float>::Sqrt(fC).z);
    CHECK(resFloat[11].w == Vector4<float>::Sqrt(fC).w);

    CHECK(resFloat[12] == Vector4<float>(12.4f, 30000.2f, 10.5f, 310.12f));
    CHECK(resFloat[17] == Vector4<float>(12.4f, 30000.2f, 10.5f, 310.12f));

    CHECK(resFloat[19] == Vector4<float>(12.4f, -30000.2f, -10.5f, -4096.9f));
    CHECK(resFloat[20] == Vector4<float>(120.1f, -3000.1f, 30.2f, 310.12f));

    CHECK(resFloat[21] == Vector4<float>(12.4f, -30000.2f, -10.5f, -4096.9f));
    CHECK(resFloat[22] == Vector4<float>(120.1f, -3000.1f, 30.2f, 310.12f));

    // Vector4<float> fA(12.4f, -30000.2f, -10.5f, 310.12f);
    CHECK(resFloat[23] == Vector4<float>(12.0f, -30000.0f, -11.0f, 310.0f));
    CHECK(resFloat[24] == Vector4<float>(13.0f, -30000.0f, -10.0f, 311.0f));
    CHECK(resFloat[25] == Vector4<float>(12.0f, -30001.0f, -11.0f, 310.0f));
    CHECK(resFloat[26] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));
    CHECK(resFloat[27] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));

    CHECK(resFloat[28] == Vector4<float>(12.0f, -30000.0f, -11.0f, 310.0f));
    CHECK(resFloat[29] == Vector4<float>(13.0f, -30000.0f, -10.0f, 311.0f));
    CHECK(resFloat[30] == Vector4<float>(12.0f, -30001.0f, -11.0f, 310.0f));
    CHECK(resFloat[31] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));
    CHECK(resFloat[32] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));

    CHECK(resFloat[33] == Vector4<float>(1.0f, 2.0f, 3.0f, 4.0f));
    CHECK(resFloat[34] == Vector4<float>(2.0f, 4.0f, 6.0f, 8.0f));

    CHECK(resFloat[35] == fB - fA);
    CHECK(resFloat[36] == fB / fA);

    CHECK(resCompFloat[0] == Vector4<float>::CompareEQ(fA, fC));
    CHECK(resCompFloat[1] == Vector4<float>::CompareLE(fA, fC));
    CHECK(resCompFloat[2] == Vector4<float>::CompareLT(fA, fC));
    CHECK(resCompFloat[3] == Vector4<float>::CompareGE(fA, fC));
    CHECK(resCompFloat[4] == Vector4<float>::CompareGT(fA, fC));
    CHECK(resCompFloat[5] == Vector4<float>::CompareNEQ(fA, fC));

    // ==
    CHECK(resCompFloat[6] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[7] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[8] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[9] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[10] == Pixel8uC4(byte(0)));
    // <=
    CHECK(resCompFloat[11] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[12] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[13] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[14] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[15] == Pixel8uC4(byte(0)));
    // <
    CHECK(resCompFloat[16] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[17] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[18] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[19] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[20] == Pixel8uC4(byte(0)));
    // >=
    CHECK(resCompFloat[21] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[22] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[23] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[24] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[25] == Pixel8uC4(byte(0)));
    // >
    CHECK(resCompFloat[26] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[27] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[28] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[29] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[30] == Pixel8uC4(byte(0)));
    // !=
    CHECK(resCompFloat[31] == Pixel8uC4(byte(0)));
    CHECK(resCompFloat[32] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[33] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[34] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[35] == Pixel8uC4(byte(1)));
    CHECK(resCompFloat[36] == Pixel8uC4(byte(1)));
}