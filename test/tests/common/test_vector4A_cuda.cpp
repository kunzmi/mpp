#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVar.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/numeric_limits.h>
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
void runtest_vector4A_kernel(Vector4A<byte> *aDataByte, Vector4A<byte> *aCompByte,         //
                             Vector4A<sbyte> *resSBytes, Vector4A<byte> *aCompSbyte,       //
                             Vector4A<short> *aDataShort, Vector4A<byte> *aCompShort,      //
                             Vector4A<ushort> *aDataUShort, Vector4A<byte> *aCompUShort,   //
                             Vector4A<BFloat16> *aDataBFloat, Vector4A<byte> *aCompBFloat, //
                             Vector4A<HalfFp16> *aDataHalf, Vector4A<byte> *aCompHalf,     //
                             Vector4A<float> *aDataFloat, Vector4A<byte> *aCompFloat);
}

TEST_CASE("Vector<short> SIMD CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    DevVar<Vector4A<sbyte>> sbytes(23);
    DevVar<Vector4A<byte>> compSBytes(37);
    DevVar<Vector4A<byte>> bytes(23);
    DevVar<Vector4A<byte>> compBytes(37);
    DevVar<Vector4A<short>> shorts(23);
    DevVar<Vector4A<byte>> compShorts(37);
    DevVar<Vector4A<ushort>> ushorts(23);
    DevVar<Vector4A<byte>> compUShorts(37);
    DevVar<Vector4A<BFloat16>> bfloats(35);
    DevVar<Vector4A<byte>> compBFloats(37);
    DevVar<Vector4A<HalfFp16>> halfs(35);
    DevVar<Vector4A<byte>> compHalfs(37);
    DevVar<Vector4A<float>> floats(35);
    DevVar<Vector4A<byte>> compFloats(37);

    runtest_vector4A_kernel(bytes.Pointer(), compBytes.Pointer(),     //
                            sbytes.Pointer(), compSBytes.Pointer(),   //
                            shorts.Pointer(), compShorts.Pointer(),   //
                            ushorts.Pointer(), compUShorts.Pointer(), //
                            bfloats.Pointer(), compBFloats.Pointer(), //
                            halfs.Pointer(), compHalfs.Pointer(),     //
                            floats.Pointer(), compFloats.Pointer());

    cudaSafeCall(cudaDeviceSynchronize());
    std::vector<Vector4A<sbyte>> resSBytes(23);
    std::vector<Vector4A<byte>> resCompSBytes(37);
    std::vector<Vector4A<byte>> resBytes(23);
    std::vector<Vector4A<byte>> resCompBytes(37);
    std::vector<Vector4A<short>> resShort(23);
    std::vector<Vector4A<byte>> resCompShort(37);
    std::vector<Vector4A<ushort>> resUShort(23);
    std::vector<Vector4A<byte>> resCompUShort(37);
    std::vector<Vector4A<BFloat16>> resBFloat(35);
    std::vector<Vector4A<byte>> resCompBFloat(37);
    std::vector<Vector4A<HalfFp16>> resHalf(35);
    std::vector<Vector4A<byte>> resCompHalf(37);
    std::vector<Vector4A<float>> resFloat(35);
    std::vector<Vector4A<byte>> resCompFloat(37);

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
    Vector4A<sbyte> sbA(12, -120, -10);
    Vector4A<sbyte> sbB(120, -80, 30);
    Vector4A<sbyte> sbC(12, 20, 0);

    CHECK(resSBytes[0] == Vector4A<sbyte>(Vector4A<float>(sbA) + Vector4A<float>(sbB)));
    CHECK(resSBytes[1] == Vector4A<sbyte>(Vector4A<float>(sbA) - Vector4A<float>(sbB)));
    CHECK(resSBytes[2] == sbA * sbB);
    CHECK(resSBytes[3] == sbA / sbB);

    CHECK(resSBytes[4] == Vector4A<sbyte>(Vector4A<float>(sbA) + Vector4A<float>(sbB)));
    CHECK(resSBytes[5] == Vector4A<sbyte>(Vector4A<float>(sbA) - Vector4A<float>(sbB)));
    CHECK(resSBytes[6] == sbA * sbB);
    CHECK(resSBytes[7] == sbA / sbB);

    CHECK(resSBytes[8] == Vector4A<sbyte>(-12, 120, 10));
    CHECK(resSBytes[12] == Vector4A<sbyte>(12, 120, 10));
    CHECK(resSBytes[13] == Vector4A<sbyte>(108, 40, 40));
    CHECK(resSBytes[17] == Vector4A<sbyte>(12, 120, 10));
    CHECK(resSBytes[18] == Vector4A<sbyte>(108, 40, 40));

    CHECK(resSBytes[19] == Vector4A<sbyte>(12, -120, -10));
    CHECK(resSBytes[20] == Vector4A<sbyte>(120, -80, 30));

    CHECK(resSBytes[21] == Vector4A<sbyte>(12, -120, -10));
    CHECK(resSBytes[22] == Vector4A<sbyte>(120, -80, 30));

    CHECK(resCompSBytes[0] == Vector4A<sbyte>::CompareEQ(sbA, sbC));
    CHECK(resCompSBytes[1] == Vector4A<sbyte>::CompareLE(sbA, sbC));
    CHECK(resCompSBytes[2] == Vector4A<sbyte>::CompareLT(sbA, sbC));
    CHECK(resCompSBytes[3] == Vector4A<sbyte>::CompareGE(sbA, sbC));
    CHECK(resCompSBytes[4] == Vector4A<sbyte>::CompareGT(sbA, sbC));
    CHECK(resCompSBytes[5] == Vector4A<sbyte>::CompareNEQ(sbA, sbC));

    // ==
    CHECK(resCompSBytes[6] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[7] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[8] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[9] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[10] == Pixel8uC4A(byte(0)));
    // <=
    CHECK(resCompSBytes[11] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[12] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[13] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[14] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[15] == Pixel8uC4A(byte(1)));
    // <
    CHECK(resCompSBytes[16] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[17] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[18] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[19] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[20] == Pixel8uC4A(byte(1)));
    // >=
    CHECK(resCompSBytes[21] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[22] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[23] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[24] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[25] == Pixel8uC4A(byte(0)));
    // >
    CHECK(resCompSBytes[26] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[27] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[28] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[29] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[30] == Pixel8uC4A(byte(0)));
    // !=
    CHECK(resCompSBytes[31] == Pixel8uC4A(byte(0)));
    CHECK(resCompSBytes[32] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[33] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[34] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[35] == Pixel8uC4A(byte(1)));
    CHECK(resCompSBytes[36] == Pixel8uC4A(byte(1)));

    // byte
    Vector4A<byte> bA(12, 120, 100);
    Vector4A<byte> bB(120, 180, 30);
    Vector4A<byte> bC(12, 20, 100);

    CHECK(resBytes[0] == Vector4A<byte>(Vector4A<float>(bA) + Vector4A<float>(bB)));
    CHECK(resBytes[1] == Vector4A<byte>(Vector4A<float>(bA) - Vector4A<float>(bB)));
    CHECK(resBytes[2] == bA * bB);
    CHECK(resBytes[3] == bA / bB);

    CHECK(resBytes[4] == Vector4A<byte>(Vector4A<float>(bA) + Vector4A<float>(bB)));
    CHECK(resBytes[5] == Vector4A<byte>(Vector4A<float>(bA) - Vector4A<float>(bB)));
    CHECK(resBytes[6] == bA * bB);
    CHECK(resBytes[7] == bA / bB);

    CHECK(resBytes[13] == Vector4A<byte>(108, 60, 70));
    CHECK(resBytes[18] == Vector4A<byte>(108, 60, 70));

    CHECK(resBytes[19] == Vector4A<byte>(12, 120, 30));
    CHECK(resBytes[20] == Vector4A<byte>(120, 180, 100));
    CHECK(resBytes[21] == Vector4A<byte>(12, 120, 30));
    CHECK(resBytes[22] == Vector4A<byte>(120, 180, 100));

    CHECK(resCompBytes[0] == Vector4A<byte>::CompareEQ(bA, bC));
    CHECK(resCompBytes[1] == Vector4A<byte>::CompareLE(bA, bC));
    CHECK(resCompBytes[2] == Vector4A<byte>::CompareLT(bA, bC));
    CHECK(resCompBytes[3] == Vector4A<byte>::CompareGE(bA, bC));
    CHECK(resCompBytes[4] == Vector4A<byte>::CompareGT(bA, bC));
    CHECK(resCompBytes[5] == Vector4A<byte>::CompareNEQ(bA, bC));

    // ==
    CHECK(resCompBytes[6] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[7] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[8] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[9] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[10] == Pixel8uC4A(byte(0)));
    // <=
    CHECK(resCompBytes[11] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[12] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[13] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[14] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[15] == Pixel8uC4A(byte(1)));
    // <
    CHECK(resCompBytes[16] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[17] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[18] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[19] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[20] == Pixel8uC4A(byte(1)));
    // >=
    CHECK(resCompBytes[21] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[22] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[23] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[24] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[25] == Pixel8uC4A(byte(0)));
    // >
    CHECK(resCompBytes[26] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[27] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[28] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[29] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[30] == Pixel8uC4A(byte(0)));
    // !=
    CHECK(resCompBytes[31] == Pixel8uC4A(byte(0)));
    CHECK(resCompBytes[32] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[33] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[34] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[35] == Pixel8uC4A(byte(1)));
    CHECK(resCompBytes[36] == Pixel8uC4A(byte(1)));

    // short
    Vector4A<short> sA(12, -30000, -10);
    Vector4A<short> sB(120, -3000, 30);
    Vector4A<short> sC(12, -30000, 0);

    CHECK(resShort[0] == Vector4A<short>(Vector4A<float>(sA) + Vector4A<float>(sB)));
    CHECK(resShort[1] == Vector4A<short>(Vector4A<float>(sA) - Vector4A<float>(sB)));
    CHECK(resShort[2] == sA * sB);
    CHECK(resShort[3] == sA / sB);

    CHECK(resShort[4] == Vector4A<short>(Vector4A<float>(sA) + Vector4A<float>(sB)));
    CHECK(resShort[5] == Vector4A<short>(Vector4A<float>(sA) - Vector4A<float>(sB)));
    CHECK(resShort[6] == sA * sB);
    CHECK(resShort[7] == sA / sB);

    CHECK(resShort[8] == Vector4A<short>(-12, 30000, 10));
    CHECK(resShort[12] == Vector4A<short>(12, 30000, 10));
    CHECK(resShort[13] == Vector4A<short>(108, 27000, 40));
    CHECK(resShort[17] == Vector4A<short>(12, 30000, 10));
    CHECK(resShort[18] == Vector4A<short>(108, 27000, 40));

    CHECK(resShort[19] == Vector4A<short>(12, -30000, -10));
    CHECK(resShort[20] == Vector4A<short>(120, -3000, 30));

    CHECK(resShort[21] == Vector4A<short>(12, -30000, -10));
    CHECK(resShort[22] == Vector4A<short>(120, -3000, 30));

    CHECK(resCompShort[0] == Vector4A<short>::CompareEQ(sA, sC));
    CHECK(resCompShort[1] == Vector4A<short>::CompareLE(sA, sC));
    CHECK(resCompShort[2] == Vector4A<short>::CompareLT(sA, sC));
    CHECK(resCompShort[3] == Vector4A<short>::CompareGE(sA, sC));
    CHECK(resCompShort[4] == Vector4A<short>::CompareGT(sA, sC));
    CHECK(resCompShort[5] == Vector4A<short>::CompareNEQ(sA, sC));

    // ==
    CHECK(resCompShort[6] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[7] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[8] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[9] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[10] == Pixel8uC4A(byte(0)));
    // <=
    CHECK(resCompShort[11] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[12] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[13] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[14] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[15] == Pixel8uC4A(byte(1)));
    // <
    CHECK(resCompShort[16] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[17] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[18] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[19] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[20] == Pixel8uC4A(byte(1)));
    // >=
    CHECK(resCompShort[21] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[22] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[23] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[24] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[25] == Pixel8uC4A(byte(0)));
    // >
    CHECK(resCompShort[26] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[27] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[28] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[29] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[30] == Pixel8uC4A(byte(0)));
    // !=
    CHECK(resCompShort[31] == Pixel8uC4A(byte(0)));
    CHECK(resCompShort[32] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[33] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[34] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[35] == Pixel8uC4A(byte(1)));
    CHECK(resCompShort[36] == Pixel8uC4A(byte(1)));

    // ushort
    Vector4A<ushort> usA(12, 60000, 100);
    Vector4A<ushort> usB(120, 7000, 30);
    Vector4A<ushort> usC(12, 20, 100);

    CHECK(resUShort[0] == Vector4A<ushort>(Vector4A<float>(usA) + Vector4A<float>(usB)));
    CHECK(resUShort[1] == Vector4A<ushort>(Vector4A<float>(usA) - Vector4A<float>(usB)));
    CHECK(resUShort[2] == usA * usB);
    CHECK(resUShort[3] == usA / usB);

    CHECK(resUShort[4] == Vector4A<ushort>(Vector4A<float>(usA) + Vector4A<float>(usB)));
    CHECK(resUShort[5] == Vector4A<ushort>(Vector4A<float>(usA) - Vector4A<float>(usB)));
    CHECK(resUShort[6] == usA * usB);
    CHECK(resUShort[7] == usA / usB);

    CHECK(resUShort[13] == Vector4A<ushort>(108, 53000, 70));
    CHECK(resUShort[18] == Vector4A<ushort>(108, 53000, 70));

    CHECK(resUShort[19] == Vector4A<ushort>(12, 7000, 30));
    CHECK(resUShort[20] == Vector4A<ushort>(120, 60000, 100));
    CHECK(resUShort[21] == Vector4A<ushort>(12, 7000, 30));
    CHECK(resUShort[22] == Vector4A<ushort>(120, 60000, 100));

    CHECK(resCompUShort[0] == Vector4A<ushort>::CompareEQ(usA, usC));
    CHECK(resCompUShort[1] == Vector4A<ushort>::CompareLE(usA, usC));
    CHECK(resCompUShort[2] == Vector4A<ushort>::CompareLT(usA, usC));
    CHECK(resCompUShort[3] == Vector4A<ushort>::CompareGE(usA, usC));
    CHECK(resCompUShort[4] == Vector4A<ushort>::CompareGT(usA, usC));
    CHECK(resCompUShort[5] == Vector4A<ushort>::CompareNEQ(usA, usC));

    // ==
    CHECK(resCompUShort[6] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[7] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[8] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[9] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[10] == Pixel8uC4A(byte(0)));
    // <=
    CHECK(resCompUShort[11] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[12] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[13] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[14] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[15] == Pixel8uC4A(byte(1)));
    // <
    CHECK(resCompUShort[16] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[17] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[18] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[19] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[20] == Pixel8uC4A(byte(1)));
    // >=
    CHECK(resCompUShort[21] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[22] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[23] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[24] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[25] == Pixel8uC4A(byte(0)));
    // >
    CHECK(resCompUShort[26] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[27] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[28] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[29] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[30] == Pixel8uC4A(byte(0)));
    // !=
    CHECK(resCompUShort[31] == Pixel8uC4A(byte(0)));
    CHECK(resCompUShort[32] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[33] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[34] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[35] == Pixel8uC4A(byte(1)));
    CHECK(resCompUShort[36] == Pixel8uC4A(byte(1)));

    // BFloat16
    Vector4A<BFloat16> bfA(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f));
    Vector4A<BFloat16> bfB(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f));
    Vector4A<BFloat16> bfC(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(0.0f));

    CHECK(resBFloat[0] == bfA + bfB);
    CHECK(resBFloat[1] == bfA - bfB);
    CHECK(resBFloat[2] == bfA * bfB);
    CHECK(resBFloat[3] == bfA / bfB);

    CHECK(resBFloat[4] == bfA + bfB);
    CHECK(resBFloat[5] == bfA - bfB);
    CHECK(resBFloat[6] == bfA * bfB);
    CHECK(resBFloat[7] == bfA / bfB);

    CHECK(resBFloat[8] == Vector4A<BFloat16>(BFloat16(-12.4f), BFloat16(30000.2f), BFloat16(10.5f)));
    CHECK(resBFloat[9] == Vector4A<BFloat16>::Exp(bfC));

    CHECK(resBFloat[10].x == Vector4A<BFloat16>::Ln(bfC).x);
    CHECK(isnan(resBFloat[10].y));
    CHECK(resBFloat[10].z == Vector4A<BFloat16>::Ln(bfC).z);

    CHECK(resBFloat[11].x == Vector4A<BFloat16>::Sqrt(bfC).x);
    CHECK(isnan(resBFloat[11].y));
    CHECK(resBFloat[11].z == Vector4A<BFloat16>::Sqrt(bfC).z);

    CHECK(resBFloat[12] == Vector4A<BFloat16>(BFloat16(12.4f), BFloat16(30000.2f), BFloat16(10.5f)));
    CHECK(resBFloat[17] == Vector4A<BFloat16>(BFloat16(12.4f), BFloat16(30000.2f), BFloat16(10.5f)));

    CHECK(resBFloat[19] == Vector4A<BFloat16>(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f)));
    CHECK(resBFloat[20] == Vector4A<BFloat16>(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f)));

    CHECK(resBFloat[21] == Vector4A<BFloat16>(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f)));
    CHECK(resBFloat[22] == Vector4A<BFloat16>(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f)));

    CHECK(resBFloat[23] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f)));
    CHECK(resBFloat[24] == Vector4A<BFloat16>(BFloat16(13.0f), BFloat16(-29952.0f), BFloat16(-10.0f)));
    CHECK(resBFloat[25] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f)));
    CHECK(resBFloat[26] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f)));
    CHECK(resBFloat[27] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f)));

    CHECK(resBFloat[28] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f)));
    CHECK(resBFloat[29] == Vector4A<BFloat16>(BFloat16(13.0f), BFloat16(-29952.0f), BFloat16(-10.0f)));
    CHECK(resBFloat[30] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f)));
    CHECK(resBFloat[31] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f)));
    CHECK(resBFloat[32] == Vector4A<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f)));

    CHECK(resBFloat[33] == Vector4A<BFloat16>(BFloat16(1.0f), BFloat16(2.0f), BFloat16(3.0f)));
    CHECK(resBFloat[34] == Vector4A<BFloat16>(BFloat16(2.0f), BFloat16(4.0f), BFloat16(6.0f)));

    CHECK(resCompBFloat[0] == Vector4A<BFloat16>::CompareEQ(bfA, bfC));
    CHECK(resCompBFloat[1] == Vector4A<BFloat16>::CompareLE(bfA, bfC));
    CHECK(resCompBFloat[2] == Vector4A<BFloat16>::CompareLT(bfA, bfC));
    CHECK(resCompBFloat[3] == Vector4A<BFloat16>::CompareGE(bfA, bfC));
    CHECK(resCompBFloat[4] == Vector4A<BFloat16>::CompareGT(bfA, bfC));
    CHECK(resCompBFloat[5] == Vector4A<BFloat16>::CompareNEQ(bfA, bfC));

    // ==
    CHECK(resCompBFloat[6] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[7] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[8] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[9] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[10] == Pixel8uC4A(byte(0)));
    // <=
    CHECK(resCompBFloat[11] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[12] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[13] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[14] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[15] == Pixel8uC4A(byte(1)));
    // <
    CHECK(resCompBFloat[16] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[17] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[18] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[19] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[20] == Pixel8uC4A(byte(1)));
    // >=
    CHECK(resCompBFloat[21] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[22] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[23] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[24] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[25] == Pixel8uC4A(byte(0)));
    // >
    CHECK(resCompBFloat[26] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[27] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[28] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[29] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[30] == Pixel8uC4A(byte(0)));
    // !=
    CHECK(resCompBFloat[31] == Pixel8uC4A(byte(0)));
    CHECK(resCompBFloat[32] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[33] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[34] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[35] == Pixel8uC4A(byte(1)));
    CHECK(resCompBFloat[36] == Pixel8uC4A(byte(1)));

    // Half
    Vector4A<HalfFp16> hA(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f));
    Vector4A<HalfFp16> hB(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f));
    Vector4A<HalfFp16> hC(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(0.0f));

    CHECK(resHalf[0] == hA + hB);
    CHECK(resHalf[1] == hA - hB);
    CHECK(resHalf[2] == hA * hB);
    CHECK(resHalf[3] == hA / hB);

    CHECK(resHalf[4] == hA + hB);
    CHECK(resHalf[5] == hA - hB);
    CHECK(resHalf[6] == hA * hB);
    CHECK(resHalf[7] == hA / hB);

    CHECK(resHalf[8] == Vector4A<HalfFp16>(HalfFp16(-12.4f), HalfFp16(30000.2f), HalfFp16(10.5f)));
    CHECK(resHalf[9] == Vector4A<HalfFp16>::Exp(hC));

    CHECK(resHalf[10].x == Vector4A<HalfFp16>::Ln(hC).x);
    CHECK(isnan(resHalf[10].y));
    CHECK(resHalf[10].z == Vector4A<HalfFp16>::Ln(hC).z);

    CHECK(resHalf[11].x == Vector4A<HalfFp16>::Sqrt(hC).x);
    CHECK(isnan(resBFloat[11].y));
    CHECK(resHalf[11].z == Vector4A<HalfFp16>::Sqrt(hC).z);

    CHECK(resHalf[12] == Vector4A<HalfFp16>(HalfFp16(12.4f), HalfFp16(30000.2f), HalfFp16(10.5f)));
    CHECK(resHalf[17] == Vector4A<HalfFp16>(HalfFp16(12.4f), HalfFp16(30000.2f), HalfFp16(10.5f)));

    CHECK(resHalf[19] == Vector4A<HalfFp16>(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f)));
    CHECK(resHalf[20] == Vector4A<HalfFp16>(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f)));

    CHECK(resHalf[21] == Vector4A<HalfFp16>(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f)));
    CHECK(resHalf[22] == Vector4A<HalfFp16>(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f)));

    CHECK(resHalf[23] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11)));
    CHECK(resHalf[24] == Vector4A<HalfFp16>(HalfFp16(13), HalfFp16(-30000), HalfFp16(-10)));
    CHECK(resHalf[25] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11)));
    CHECK(resHalf[26] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10)));
    CHECK(resHalf[27] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10)));

    CHECK(resHalf[28] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11)));
    CHECK(resHalf[29] == Vector4A<HalfFp16>(HalfFp16(13), HalfFp16(-30000), HalfFp16(-10)));
    CHECK(resHalf[30] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11)));
    CHECK(resHalf[31] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10)));
    CHECK(resHalf[32] == Vector4A<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10)));

    CHECK(resHalf[33] == Vector4A<HalfFp16>(HalfFp16(1), HalfFp16(2), HalfFp16(3)));
    CHECK(resHalf[34] == Vector4A<HalfFp16>(HalfFp16(2), HalfFp16(4), HalfFp16(6)));

    CHECK(resCompHalf[0] == Vector4A<HalfFp16>::CompareEQ(hA, hC));
    CHECK(resCompHalf[1] == Vector4A<HalfFp16>::CompareLE(hA, hC));
    CHECK(resCompHalf[2] == Vector4A<HalfFp16>::CompareLT(hA, hC));
    CHECK(resCompHalf[3] == Vector4A<HalfFp16>::CompareGE(hA, hC));
    CHECK(resCompHalf[4] == Vector4A<HalfFp16>::CompareGT(hA, hC));
    CHECK(resCompHalf[5] == Vector4A<HalfFp16>::CompareNEQ(hA, hC));

    // ==
    CHECK(resCompHalf[6] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[7] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[8] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[9] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[10] == Pixel8uC4A(byte(0)));
    // <=
    CHECK(resCompHalf[11] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[12] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[13] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[14] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[15] == Pixel8uC4A(byte(1)));
    // <
    CHECK(resCompHalf[16] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[17] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[18] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[19] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[20] == Pixel8uC4A(byte(1)));
    // >=
    CHECK(resCompHalf[21] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[22] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[23] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[24] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[25] == Pixel8uC4A(byte(0)));
    // >
    CHECK(resCompHalf[26] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[27] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[28] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[29] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[30] == Pixel8uC4A(byte(0)));
    // !=
    CHECK(resCompHalf[31] == Pixel8uC4A(byte(0)));
    CHECK(resCompHalf[32] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[33] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[34] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[35] == Pixel8uC4A(byte(1)));
    CHECK(resCompHalf[36] == Pixel8uC4A(byte(1)));

    // Float (32bit)
    Vector4A<float> fA(12.4f, -30000.2f, -10.5f);
    Vector4A<float> fB(120.1f, -3000.1f, 30.2f);
    Vector4A<float> fC(12.4f, -30000.2f, 0.0f);

    CHECK(resFloat[0] == fA + fB);
    CHECK(resFloat[1] == fA - fB);
    CHECK(resFloat[2] == fA * fB);
    CHECK(resFloat[3] == fA / fB);

    CHECK(resFloat[4] == fA + fB);
    CHECK(resFloat[5] == fA - fB);
    CHECK(resFloat[6] == fA * fB);
    CHECK(resFloat[7] == fA / fB);

    CHECK(resFloat[8] == Vector4A<float>(-12.4f, 30000.2f, 10.5f));
    // GCC slighlty differs for exact comparison:
    CHECK(resFloat[9].x == Approx(Vector4A<float>::Exp(fC).x).margin(0.001));
    CHECK(resFloat[9].y == Vector4A<float>::Exp(fC).y);
    CHECK(resFloat[9].z == Vector4A<float>::Exp(fC).z);

    CHECK(resFloat[10].x == Vector4A<float>::Ln(fC).x);
    CHECK(isnan(resFloat[10].y));
    CHECK(resFloat[10].z == Vector4A<float>::Ln(fC).z);

    CHECK(resFloat[11].x == Vector4A<float>::Sqrt(fC).x);
    CHECK(isnan(resFloat[11].y));
    CHECK(resFloat[11].z == Vector4A<float>::Sqrt(fC).z);

    CHECK(resFloat[12] == Vector4A<float>(12.4f, 30000.2f, 10.5f));
    CHECK(resFloat[17] == Vector4A<float>(12.4f, 30000.2f, 10.5f));

    CHECK(resFloat[19] == Vector4A<float>(12.4f, -30000.2f, -10.5f));
    CHECK(resFloat[20] == Vector4A<float>(120.1f, -3000.1f, 30.2f));

    CHECK(resFloat[21] == Vector4A<float>(12.4f, -30000.2f, -10.5f));
    CHECK(resFloat[22] == Vector4A<float>(120.1f, -3000.1f, 30.2f));

    CHECK(resFloat[23] == Vector4A<float>(12.0f, -30000.0f, -11.0f));
    CHECK(resFloat[24] == Vector4A<float>(13.0f, -30000.0f, -10.0f));
    CHECK(resFloat[25] == Vector4A<float>(12.0f, -30001.0f, -11.0f));
    CHECK(resFloat[26] == Vector4A<float>(12.0f, -30000.0f, -10.0f));
    CHECK(resFloat[27] == Vector4A<float>(12.0f, -30000.0f, -10.0f));

    CHECK(resFloat[28] == Vector4A<float>(12.0f, -30000.0f, -11.0f));
    CHECK(resFloat[29] == Vector4A<float>(13.0f, -30000.0f, -10.0f));
    CHECK(resFloat[30] == Vector4A<float>(12.0f, -30001.0f, -11.0f));
    CHECK(resFloat[31] == Vector4A<float>(12.0f, -30000.0f, -10.0f));
    CHECK(resFloat[32] == Vector4A<float>(12.0f, -30000.0f, -10.0f));

    CHECK(resFloat[33] == Vector4A<float>(1.0f, 2.0f, 3.0f));
    CHECK(resFloat[34] == Vector4A<float>(2.0f, 4.0f, 6.0f));

    CHECK(resCompFloat[0] == Vector4A<float>::CompareEQ(fA, fC));
    CHECK(resCompFloat[1] == Vector4A<float>::CompareLE(fA, fC));
    CHECK(resCompFloat[2] == Vector4A<float>::CompareLT(fA, fC));
    CHECK(resCompFloat[3] == Vector4A<float>::CompareGE(fA, fC));
    CHECK(resCompFloat[4] == Vector4A<float>::CompareGT(fA, fC));
    CHECK(resCompFloat[5] == Vector4A<float>::CompareNEQ(fA, fC));

    // ==
    CHECK(resCompFloat[6] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[7] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[8] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[9] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[10] == Pixel8uC4A(byte(0)));
    // <=
    CHECK(resCompFloat[11] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[12] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[13] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[14] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[15] == Pixel8uC4A(byte(1)));
    // <
    CHECK(resCompFloat[16] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[17] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[18] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[19] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[20] == Pixel8uC4A(byte(1)));
    // >=
    CHECK(resCompFloat[21] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[22] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[23] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[24] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[25] == Pixel8uC4A(byte(0)));
    // >
    CHECK(resCompFloat[26] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[27] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[28] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[29] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[30] == Pixel8uC4A(byte(0)));
    // !=
    CHECK(resCompFloat[31] == Pixel8uC4A(byte(0)));
    CHECK(resCompFloat[32] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[33] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[34] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[35] == Pixel8uC4A(byte(1)));
    CHECK(resCompFloat[36] == Pixel8uC4A(byte(1)));
}